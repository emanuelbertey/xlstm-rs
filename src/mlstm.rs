/*
# mLSTM: Matrix Long Short-Term Memory

This module implements the mLSTM (matrix LSTM) cell and layer as described in the paper:
"xLSTM: Extended Long Short-Term Memory" by Beck et al. (2024).

The mLSTM extends the traditional LSTM by using a matrix memory state and exponential gating,
allowing for enhanced storage capacities and improved performance on long-range dependencies.
*/

use burn::{
    config::Config,
    module::{Module, Param},
    nn::{Dropout, DropoutConfig, Initializer, Linear, LinearConfig},
    tensor::{Tensor, activation, backend::Backend},
};
use num_traits::{FromPrimitive, ToPrimitive};

/// State for mLSTM containing cell matrix and hidden state
#[derive(Clone, Debug)]
pub struct MLstmstate<B: Backend> {
    /// Cell state - matrix of shape [`batch_size`, `num_heads`, `head_dim`, `head_dim`]
    pub cell: Tensor<B, 4>,
    /// Hidden state - vector of shape [`batch_size`, `hidden_size`]
    pub hidden: Tensor<B, 2>,
    /// Normalizer state - vector of shape [`batch_size`, `num_heads`, `head_dim`]
    pub normalizer: Tensor<B, 3>,
    /// Global max gate state for numeric stability - shape [`batch_size`, `num_heads`, 1]
    pub max_gate_log: Tensor<B, 3>,
}

impl<B: Backend> MLstmstate<B> {
    /// Create a new mLSTM state
    pub const fn new(
        cell: Tensor<B, 4>,
        hidden: Tensor<B, 2>,
        normalizer: Tensor<B, 3>,
        max_gate_log: Tensor<B, 3>,
    ) -> Self {
        Self {
            cell,
            hidden,
            normalizer,
            max_gate_log,
        }
    }

    /// Detach the state from the computational graph
    pub fn detach(self) -> Self {
        Self {
            cell: self.cell.detach(),
            hidden: self.hidden.detach(),
            normalizer: self.normalizer.detach(),
            max_gate_log: self.max_gate_log.detach(),
        }
    }
}

/// Configuration for mLSTM
#[derive(Config, Debug)]
pub struct MLstmconfig {
    /// Size of input features
    pub d_input: usize,
    /// Size of hidden state
    pub d_hidden: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Number of heads for multi-head mLSTM
    #[config(default = "4")]
    pub num_heads: usize,
    /// Dropout probability
    #[config(default = "0.0")]
    pub dropout: f64,
    /// Weight initializer
    #[config(default = "Initializer::XavierNormal{gain:1.0}")]
    pub initializer: Initializer,
}
impl MLstmconfig {
    /// Initialize a new mLSTM
    pub fn init<B: Backend>(&self, device: &B::Device) -> MLstm<B> {
        let layers = (0..self.num_layers)
            .map(|i| {
                let input_size = if i == 0 { self.d_input } else { self.d_hidden };
                MLstmcell::new(input_size, self.d_hidden, self.num_heads, &self.initializer, device)
            })
            .collect();

        MLstm {
            layers,
            dropout_layer: DropoutConfig::new(self.dropout).init(),
            d_input: self.d_input,
            d_hidden: self.d_hidden,
            num_layers: self.num_layers,
            num_heads: self.num_heads,
            dropout: self.dropout,
        }
    }
}

/// mLSTM layer implementation
#[derive(Module, Debug)]
pub struct MLstm<B: Backend> {
    /// Stack of mLSTM cells
    pub layers: alloc::vec::Vec<MLstmcell<B>>,
    /// Dropout module for inter-layer dropout
    pub dropout_layer: Dropout,
    /// Input size
    pub d_input: usize,
    /// Hidden size
    pub d_hidden: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Number of heads
    pub num_heads: usize,
    /// Dropout probability
    pub dropout: f64,
}

impl<B: Backend> MLstm<B> {
    /// Forward pass through mLSTM consuming and returning states
    ///
    /// # Arguments
    /// * `input_seq` - Input tensor of shape [`batch_size`, `seq_length`, `input_size`]
    /// * `states` - States to consume (will be moved)
    ///
    /// # Returns
    /// * Output tensor of shape [`batch_size`, `seq_length`, `hidden_size`]
    /// * New states
    pub fn forward(
        &self,
        input_seq: &Tensor<B, 3>,
        states: Option<alloc::vec::Vec<MLstmstate<B>>>,
    ) -> (Tensor<B, 3>, alloc::vec::Vec<MLstmstate<B>>)
    where
        <B as Backend>::FloatElem: ToPrimitive + FromPrimitive,
    {
        let device = input_seq.device();
        let [batch_size, _seq_length, _] = input_seq.dims();

        // Inicializar estados
        let mut hidden_states = states.unwrap_or_else(|| self.init_hidden(batch_size, &device));
        let mut layer_input = input_seq.clone();

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // mLSTM processes the entire sequence using the parallel kernel (Dual Form)
            let old_state = hidden_states[layer_idx].clone();
            
            let (h_seq, new_state) = layer.forward_sequence(&layer_input, old_state);
            
            // Store the final state for future sequences (re-injecting continuity)
            hidden_states[layer_idx] = new_state;

            // Inter-layer Dropout
            layer_input = if layer_idx < self.num_layers - 1 && self.dropout > 0.0 {
                self.dropout_layer.forward(h_seq)
            } else {
                h_seq
            };
        }

        (layer_input, hidden_states)
    }

    /// Initialize hidden states
    fn init_hidden(&self, batch_size: usize, device: &B::Device) -> alloc::vec::Vec<MLstmstate<B>> {
        let head_dim = self.d_hidden / self.num_heads;
        
        (0..self.num_layers)
            .map(|_| {
                MLstmstate::new(
                    Tensor::zeros([batch_size, self.num_heads, head_dim, head_dim], device),
                    Tensor::zeros([batch_size, self.d_hidden], device),
                    Tensor::zeros([batch_size, self.num_heads, head_dim], device),
                    Tensor::zeros([batch_size, self.num_heads, 1], device),
                )
            })
            .collect()
    }
}

/// mLSTM cell implementation with matrix memory
#[derive(Module, Debug)]
pub struct MLstmcell<B: Backend> {
    /// Weight matrix for input to gates
    pub weight_ih: Param<Tensor<B, 2>>,
    /// Bias for gates
    pub bias: Param<Tensor<B, 1>>,
    /// Query projection
    pub w_q: Linear<B>,
    /// Key projection
    pub w_k: Linear<B>,
    /// Value projection
    pub w_v: Linear<B>,
    /// Input size
    pub input_size: usize,
    /// Hidden size
    pub hidden_size: usize,
    /// Number of heads
    pub num_heads: usize,
}

impl<B: Backend> MLstmcell<B> {
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        num_heads: usize,
        initializer: &Initializer,
        device: &B::Device,
    ) -> Self {
        let mut bias_data = alloc::vec![0.0; 3 * num_heads];
        for i in 0..num_heads {
            bias_data[i] = -2.0;             // Input gate: Prevent normalizer explosion
            bias_data[i + num_heads] = 0.0;  // Forget gate: Neutral (exp(0) = 1)
            bias_data[i + 2 * num_heads] = 1.0; // Output gate: Mostly open
        }
        let bias = Tensor::from_floats(bias_data.as_slice(), device);

        // Initialize Q, K, V with standard gain (1.0)
        let w_q = LinearConfig::new(input_size, hidden_size)
            .with_bias(false)
            .with_initializer(Initializer::XavierNormal { gain: 1.0 })
            .init(device);
        let w_k = LinearConfig::new(input_size, hidden_size)
            .with_bias(false)
            .with_initializer(Initializer::XavierNormal { gain: 1.0 })
            .init(device);
        let w_v = LinearConfig::new(input_size, hidden_size)
            .with_bias(false)
            .with_initializer(Initializer::XavierNormal { gain: 1.0 })
            .init(device);
        let weight_ih = initializer.init_with(
            [3 * num_heads, input_size],
            Some(input_size),
            Some(3 * num_heads),
            device,
        );

        Self {
            weight_ih,
            bias: Param::from_tensor(bias),
            w_q,
            w_k,
            w_v,
            input_size,
            hidden_size,
            num_heads,
        }
    }

    /// Forward pass through mLSTM cell consuming the state
    ///
    /// # Arguments
    /// * `input` - Input tensor [`batch_size`, `input_size`]
    /// * `state` - State to consume (moved)
    ///
    /// # Returns
    /// * New hidden state (for output)
    /// * New complete state
    pub fn forward_sequence(
        &self,
        input_seq: &Tensor<B, 3>,
        state: MLstmstate<B>,
    ) -> (Tensor<B, 3>, MLstmstate<B>)
    where
        <B as Backend>::FloatElem: num_traits::ToPrimitive + num_traits::FromPrimitive + Copy,
    {
        let [batch_size, seq_len, _] = input_seq.dims();
        let head_dim = self.hidden_size / self.num_heads;
        let device = input_seq.device();

        // 1. Parallel Projections (Q, K, V)
        let q = self.w_q.forward(input_seq.clone())
            .reshape::<4, _>([batch_size, seq_len, self.num_heads, head_dim])
            .swap_dims(1, 2); // [B, H, S, D_h]
        let k = self.w_k.forward(input_seq.clone())
            .reshape::<4, _>([batch_size, seq_len, self.num_heads, head_dim])
            .swap_dims(1, 2);
        let v = self.w_v.forward(input_seq.clone())
            .reshape::<4, _>([batch_size, seq_len, self.num_heads, head_dim])
            .swap_dims(1, 2);



        // 2. Parallel Gates (Scalar per head)
        let weight_ih_val = self.weight_ih.val().transpose();
        let bias_val = self.bias.val();

        // Proyección directa a num_heads
        let gates = input_seq.clone().matmul(weight_ih_val.reshape::<3, _>([1, self.input_size, 3 * self.num_heads])) 
                    + bias_val.reshape::<3, _>([1, 1, 3 * self.num_heads]);
        
        let i_log = gates.clone().slice([0..batch_size, 0..seq_len, 0..self.num_heads]).swap_dims(1, 2).reshape::<4, _>([batch_size, self.num_heads, seq_len, 1]); // [B, H, S, 1]
        let f_log = gates.clone().slice([0..batch_size, 0..seq_len, self.num_heads..2*self.num_heads]).swap_dims(1, 2).reshape::<4, _>([batch_size, self.num_heads, seq_len, 1]);
        let o_gate = activation::sigmoid(gates.clone().slice([0..batch_size, 0..seq_len, 2*self.num_heads..3*self.num_heads]))
            .swap_dims(1, 2)
            .reshape::<4, _>([batch_size, self.num_heads, seq_len, 1]); // [B, H, S, 1]

        // Forget gate log-space stable (PAPER ACCURATE: pure log-projection)
        let f_log_val = f_log; 
        
        // 3. Parallel Stabilization Logic (Log-Space) - PAPER ACCURATE (10/10)
        // i_log y f_log ya son escalares por cabeza [B, H, S, 1]
        let i_log_scalar = i_log; 
        let f_log_scalar = f_log_val; 
        
        // Manual CumSum using triangular matrix: [1, 1, S, S] @ [B, H, S, 1] -> [B, H, S, 1]
        let mask_tri = Tensor::<B, 2>::tril(Tensor::ones([seq_len, seq_len], &device), 0);
        let f_log_cumsum = mask_tri.clone().reshape::<4, _>([1, 1, seq_len, seq_len]).matmul(f_log_scalar.clone());
        
        // Matrix of decay weights in 4D: [B, H, S_t, S_k]
        // log_f_matrix[t, k] = sum_{j=k+1}^t log f_j = F[t] - F[k]
        let f_t = f_log_cumsum.clone(); // [B, H, S, 1]
        let f_k = f_log_cumsum.clone().swap_dims(2, 3); // [B, H, 1, S]
        let log_f_matrix = f_t - f_k;
        
        // log_weights[t, k] = log_f_matrix + i_log[k]
        let i_k = i_log_scalar.clone().swap_dims(2, 3); // [B, H, 1, S]
        let log_weights = log_f_matrix + i_k;
        
        // Causal Masking
        let mask_4d = mask_tri.reshape::<4, _>([1, 1, seq_len, seq_len]);
        let log_weights_masked = log_weights.mask_fill(mask_4d.equal(Tensor::zeros([1, 1, seq_len, seq_len], &device)), -1e30);
        
        // Contribución del estado inicial: m_0 + sum log f
        let m_0 = state.max_gate_log.clone().reshape::<4, _>([batch_size, self.num_heads, 1, 1]); 
        let log_initial_contrib = f_log_cumsum.clone() + m_0; // [B, H, S, 1]
        
        // m_t global = max( max_k(weights), log_initial_contrib )
        let max_seq = log_weights_masked.clone().max_dim(3); // [B, H, S, 1]
        let m_t_global = max_seq.max_pair(log_initial_contrib.clone()); // [B, H, S, 1]
        
        // Exponenciales estables (Escalares por cabeza)
        let weights = (log_weights_masked - m_t_global.clone()).exp(); // [B, H, S, S]
        let initial_scale = (log_initial_contrib - m_t_global.clone()).exp(); // [B, H, S, 1]
        
        // --- Compute H and N ---
        
        // H_parallel = (weights @ V)  -- wait, weights is [t, k]
        // We need sum_k weights[t, k] * (v[k] * k[k]^T) ? No. 
        // Standard attention form: H = (weights @ (K^T \odot V))? No.
        // xLSTM Matrix memory:
        // C_t = sum (v_k * k_k^T * decay)
        // h_t = q_t * C_t = sum (q_t * v_k * k_k^T * decay)
        //     = sum ((q_t * v_k^T) * k_k)? No. 
        //     = sum (scalar(q_t, k_k) * v_k) ? No, that's regular attention.
        //     = q_t * (sum v_k k_k^T) 
        //     = sum (q_t k_k) v_k ? No, matrix multiply order.
        //     = q_t * V * K^T ?
        // Let's check dimensions.
        // q: [B, H, S, D], k: [B, H, S, D], v: [B, H, S, D]
        // C is [D, D].
        // C = K^T * V ? ([D, S] * [S, D] -> [D, D])
        // h = Q * C = Q * K^T * V?
        // Assoc: (Q K^T) V. Yes.
        // So we compute Attention(Q, K, V) with our specific decay weights.
        // weights[t, k] acts as the "attention score" A[t, k].
        
        // --- Numerator (C_t @ q_t) ---
        // h_parallel = sum_k weights[t, k] * (q_t @ k_k^T) * v_k
       
       /*
        // 1. Producto punto q * k_k^T para todas las combinaciones t, k
        let qk = q.clone().matmul(k.clone().swap_dims(2, 3)); // [B, H, S, S]
        */
        // 1. Producto punto q * k_k^T con escalado de estabilidad
        let head_dim_f = head_dim as f32;
        let scale = 1.0 / head_dim_f.sqrt(); 

        let qk = q.clone().matmul(k.clone().swap_dims(2, 3)) * scale; // Standard attention scores

        // 2. Aplicamos los pesos de decaimiento escalares a las puntuaciones de atención
        // baseline let attention_scores = weights.clone() * qk.clone(); // [B, H, S, S]
        let attention_scores = weights.clone() * qk; // [B, H, S, S]
       
       
        // 3. Resultado final con valores v
        let h_parallel = attention_scores.clone().matmul(v.clone()); // [B, H, S, D]
        
        // 4. Contribución del estado inicial (PAPER ACCURATE)
        // h_0 = weights_initial * (C_0 @ q_t)
        // q es [B, H, S, D], Cell es [B, H, D, D].
        // Matmul(q, Cell.transpose) -> [B, H, S, D]
       // baseline  let h_initial = q.clone().matmul(state.cell.clone().swap_dims(2, 3)) * initial_scale.clone();
        let h_initial = (q.clone() * scale).matmul(state.cell.clone().swap_dims(2, 3)) * initial_scale.clone();
        // --- Denominator (n_t^T @ q_t) ---
        // n_parallel = sum_k weights[t, k] * k_k
        let n_parallel = weights.clone().matmul(k.clone()); // [B, H, S, D]
        let n_dot_q_parallel = (n_parallel * q.clone()).sum_dim(3).reshape::<4, _>([batch_size, self.num_heads, seq_len, 1]); // [B, H, S, 1]
        
       /* // n_initial_dot_q = weights_initial * (q @ n_0)
        let n_initial_dot_q = (q.clone() * state.normalizer.clone().reshape::<4, _>([batch_size, self.num_heads, 1, head_dim]))
            .sum_dim(3).reshape::<4, _>([batch_size, self.num_heads, seq_len, 1]) * initial_scale.clone();
        
        let denominator = n_dot_q_parallel + n_initial_dot_q;
        */
       // n_initial_dot_q = weights_initial * ( (q * scale) @ n_0 )
        let n_initial_dot_q = ((q.clone() * scale) * state.normalizer.clone().reshape::<4, _>([batch_size, self.num_heads, 1, head_dim]))
        .sum_dim(3)
        .reshape::<4, _>([batch_size, self.num_heads, seq_len, 1]) * initial_scale.clone();

        let denominator = (n_dot_q_parallel * scale) + n_initial_dot_q;

        // Estabilización final escalar (PAPER ACCURATE: max(|n^T q|, 1))
        /*// Debug: Quitamos el max(1) que pisa la señal y usamos epsilon
        let epsilon = 1e-6;
        let denominator_stable = denominator.abs() + epsilon; 

        let h_normalized = (h_parallel + h_initial) / denominator_stable;*/

        let ones = Tensor::ones_like(&denominator);
        let denominator_stable = denominator.abs().max_pair(ones); 
        
        let h_normalized = (h_parallel + h_initial) / denominator_stable;
        
        // --- Output Gate (PAPER ACCURATE per head) ---
        // h_gated = h_normalized * o_gate -> [B, H, S, D] * [B, H, S, 1]
        let h_gated = h_normalized * o_gate;
        
        // Recombinar cabezas para la salida final: [B, H, S, D] -> [B, S, Hidden]
        let h_seq = h_gated.swap_dims(1, 2).reshape::<3, _>([batch_size, seq_len, self.hidden_size]);

        // --- State Update for Next Step (FAITHFUL TO PAPER) ---
        let last_idx = seq_len - 1;
        
        // 1. m_T
        let final_m = m_t_global.slice([0..batch_size, 0..self.num_heads, last_idx..seq_len, 0..1]).reshape::<3, _>([batch_size, self.num_heads, 1]);
        
        // 2. n_T = exp(F_T - m_T) * n_0 + sum_k (weights[T, k] * k_k)
        let last_scale = initial_scale.slice([0..batch_size, 0..self.num_heads, last_idx..seq_len, 0..1]).reshape::<3, _>([batch_size, self.num_heads, 1]);
        
        // Ajuste de normalizador inicial si el batch cambió
        let s_norm = if state.normalizer.dims()[0] != batch_size {
            state.normalizer.clone().slice([0..batch_size, 0..self.num_heads, 0..head_dim])
        } else {
            state.normalizer.clone()
        };
        let n_initial_contrib = s_norm * last_scale.clone().reshape::<3, _>([batch_size, self.num_heads, 1]);
        
        let last_weights = weights.slice([0..batch_size, 0..self.num_heads, last_idx..seq_len, 0..seq_len]); 
        let n_parallel_contrib = last_weights.clone().matmul(k.clone()).reshape::<3, _>([batch_size, self.num_heads, head_dim]);
        let final_norm = n_initial_contrib + n_parallel_contrib;

        // 3. C_T = exp(F_T - m_T) * C_0 + sum_k (weights[T, k] * v_k @ k_k^T)
        // Ajuste de celda inicial si el batch cambió
        let s_cell = if state.cell.dims()[0] != batch_size {
            state.cell.clone().slice([0..batch_size, 0..self.num_heads, 0..head_dim, 0..head_dim])
        } else {
            state.cell.clone()
        };
        let c_initial_contrib = s_cell * last_scale.clone().reshape::<4, _>([batch_size, self.num_heads, 1, 1]);
        
        // sum_k weights[T, k] * (v_k @ k_k^T) --> (v_weighted^T @ k)
        let v_weighted = v * last_weights.reshape::<4, _>([batch_size, self.num_heads, seq_len, 1]);
        let c_parallel_contrib = v_weighted.swap_dims(2, 3).matmul(k); 
        let final_cell = c_initial_contrib + c_parallel_contrib;

        (h_seq.clone(), MLstmstate::new(final_cell, h_seq.slice([0..batch_size, last_idx..seq_len, 0..self.hidden_size]).reshape([batch_size, self.hidden_size]), final_norm, final_m))
    }
// no se usa no usarlo 
    fn get_causal_mask(&self, seq_len: usize, device: &B::Device) -> Tensor<B, 4, burn::tensor::Bool> {
        let indices = Tensor::<B, 1, burn::tensor::Int>::arange(0..seq_len as i64, device);
        let row_indices = indices.clone().reshape::<2, _>([seq_len, 1]).expand::<2, _>([seq_len, seq_len]);
        let col_indices = indices.reshape::<2, _>([1, seq_len]).expand::<2, _>([seq_len, seq_len]);
        col_indices.greater(row_indices).reshape::<4, _>([1, 1, seq_len, seq_len])
    }
//
    pub fn forward(
        &self,
        input: &Tensor<B, 2>,
        state: MLstmstate<B>,
    ) -> (Tensor<B, 2>, MLstmstate<B>)
    where
        <B as Backend>::FloatElem: num_traits::ToPrimitive + num_traits::FromPrimitive + Copy,
    {
        let MLstmstate { cell, hidden: _, normalizer, max_gate_log } = state;
        let [batch_size, _] = input.dims();
        let head_dim = self.hidden_size / self.num_heads;
        let _device = input.device();

        // Gates calculation (Scalar projections per head)
        let gates = input.clone().matmul(self.weight_ih.val().transpose())
            + self.bias.val().reshape::<2, _>([1, 3 * self.num_heads]);

        let chunks = gates.chunk(3, 1);
        let i_log = chunks[0].clone().reshape::<2, _>([batch_size, self.num_heads]);
        let f_log = chunks[1].clone().reshape::<2, _>([batch_size, self.num_heads]);
        let o_gate = activation::sigmoid(chunks[2].clone()).reshape::<3, _>([batch_size, self.num_heads, 1]); // [B, H, 1]

        // Projections
        let q = self.w_q.forward(input.clone()).reshape::<4, _>([batch_size, self.num_heads, 1, head_dim]);
        let k = self.w_k.forward(input.clone()).reshape::<4, _>([batch_size, self.num_heads, 1, head_dim]);
        let v = self.w_v.forward(input.clone()).reshape::<4, _>([batch_size, self.num_heads, head_dim, 1]);

        let m_t_minus_1 = max_gate_log.reshape::<2, _>([batch_size, self.num_heads]); 
        let i_log_m = i_log; 
        let f_log_m = f_log; // PAPER ACCURATE: pure log-projection
        let m_t = (f_log_m.clone() + m_t_minus_1.clone()).max_pair(i_log_m.clone()); 
        
        // Stable updates for cell and normalizer
        let f_stable = (f_log_m + m_t_minus_1 - m_t.clone()).exp();
        let i_stable = (i_log_m - m_t.clone()).exp();
        
        // Updates
        let f_exp = f_stable.clone().reshape::<4, _>([batch_size, self.num_heads, 1, 1]).expand([batch_size, self.num_heads, head_dim, head_dim]); 
        let i_exp = i_stable.clone().reshape::<4, _>([batch_size, self.num_heads, 1, 1]).expand([batch_size, self.num_heads, head_dim, head_dim]);

        let cell_update = v.clone().matmul(k.clone());
        let c_new = cell * f_exp + cell_update * i_exp;
        
        let n_new = normalizer * f_stable.clone().reshape::<3, _>([batch_size, self.num_heads, 1]).expand([batch_size, self.num_heads, head_dim]) 
                  + k.reshape::<3, _>([batch_size, self.num_heads, head_dim]) * i_stable.clone().reshape::<3, _>([batch_size, self.num_heads, 1]).expand([batch_size, self.num_heads, head_dim]);

        // Numerador: h_tilde = (q_t * scale) @ C_t^T
        let head_dim_f = head_dim as f32;
        let scale = 1.0 / head_dim_f.sqrt();
        let h_heads = (q.clone() * scale).matmul(c_new.clone().swap_dims(2, 3)).squeeze::<3>(2); // [B, H, D]
        
        // Denominador escalar (Paper Eq. 13): n_t^T * (q_t * scale)
        let q_step = (q.clone() * scale).reshape::<3, _>([batch_size, self.num_heads, head_dim]);
        let denominator = (n_new.clone() * q_step).sum_dim(2).reshape([batch_size, self.num_heads, 1]);
        
        // Denominador estable (max(|n^T q|, 1))
        let ones = Tensor::ones_like(&denominator);
        let denominator_stable = denominator.abs().max_pair(ones);

        let h_normalized = h_heads / denominator_stable;
        
        // --- Output Gate (PAPER ACCURATE per head) ---
        // h_gated = h_normalized * o_gate -> [B, H, D] * [B, H, 1]
        let h_gated = h_normalized * o_gate;
        
        // Recombinar cabezas para la salida final: [B, H, D] -> [B, Hidden]
        let h_new = h_gated.reshape::<2, _>([batch_size, self.hidden_size]);

        let new_state = MLstmstate::new(c_new, h_new.clone(), n_new, m_t.reshape::<3, _>([batch_size, self.num_heads, 1]));
        (h_new, new_state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;

    type TestBackend = burn_ndarray::NdArray<f32>;

    #[test]
    fn test_mlstm_forward() {
        let device = Default::default();
        let config = MLstmconfig::new(64, 128, 2).with_dropout(0.1);
        let mlstm = config.init::<TestBackend>(&device);

        let input = Tensor::<TestBackend, 3>::random([4, 10, 64], Distribution::Default, &device);

        let (output, states) = mlstm.forward(&input, None);

        assert_eq!(output.dims(), [4, 10, 128]);
        assert_eq!(states.len(), 2);
        assert_eq!(states[0].hidden.dims(), [4, 128]);
        assert_eq!(states[0].cell.dims(), [4, 4, 32, 32]);
    }

    #[test]
    fn test_mlstm_cell() {
        let device = Default::default();
        let num_heads = 4;
        let cell = MLstmcell::new(32, 64, num_heads, &Initializer::XavierNormal { gain: 1.0 }, &device);

        let input = Tensor::<TestBackend, 2>::random([4, 32], Distribution::Default, &device);
        let state = MLstmstate::new(
            Tensor::<TestBackend, 4>::zeros([4, num_heads, 16, 16], &device),
            Tensor::<TestBackend, 2>::zeros([4, 64], &device),
            Tensor::<TestBackend, 3>::zeros([4, num_heads, 16], &device),
            Tensor::<TestBackend, 3>::zeros([4, num_heads, 1], &device),
        );

        let (h_new, new_state) = cell.forward(&input, state);

        assert_eq!(h_new.dims(), [4, 64]);
        assert_eq!(new_state.cell.dims(), [4, 4, 16, 16]);
        assert_eq!(new_state.hidden.dims(), [4, 64]);
    }

    #[test]
    fn test_mlstm_state_reuse() {
        let device = Default::default();
        let config = MLstmconfig::new(32, 64, 1);
        let mlstm = config.init::<TestBackend>(&device);

        let input1 = Tensor::<TestBackend, 3>::random([2, 5, 32], Distribution::Default, &device);
        let input2 = Tensor::<TestBackend, 3>::random([2, 5, 32], Distribution::Default, &device);

        // First forward pass
        let (output1, states) = mlstm.forward(&input1, None);

        // Second forward pass reusing states
        let (output2, _final_states) = mlstm.forward(&input2, Some(states));

        assert_eq!(output1.dims(), [2, 5, 64]);
        assert_eq!(output2.dims(), [2, 5, 64]);
    }
}