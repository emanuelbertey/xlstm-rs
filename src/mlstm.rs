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
    nn::{Dropout, DropoutConfig, Initializer, LayerNorm, LayerNormConfig, Linear, LinearConfig},
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
    /// Weight matrix for hidden to gates
    pub weight_hh: Param<Tensor<B, 2>>,
    /// Bias for gates
    pub bias: Param<Tensor<B, 1>>,
    /// Query projection
    pub w_q: Linear<B>,
    /// Key projection
    pub w_k: Linear<B>,
    /// Value projection
    pub w_v: Linear<B>,
    /// LayerNorm for multi-head normalization
    pub ln: LayerNorm<B>,
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
        let weight_ih = initializer.init_with(
            [3 * hidden_size, input_size],
            Some(input_size),
            Some(3 * hidden_size),
            device,
        );
        // weight_hh se mantiene para compatibilidad con predict_last, 
        // pero el kernel paralelo lo ignora para permitir O(1) en tiempo.
        let weight_hh = initializer.init_with(
            [3 * hidden_size, hidden_size],
            Some(hidden_size),
            Some(3 * hidden_size),
            device,
        );

        let mut bias_data = alloc::vec![0.0; 3 * hidden_size];
        for i in 0..hidden_size {
            bias_data[i] = 0.0; // Input gate bias
        }
        for i in hidden_size..(2 * hidden_size) {
            bias_data[i] = 1.0; // Forget gate bias (increased to 1.0 for better initial memory)
        }
        let bias = Tensor::from_floats(bias_data.as_slice(), device);

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

        let head_dim = hidden_size / num_heads;
        let ln = LayerNormConfig::new(head_dim).init(device);

        Self {
            weight_ih,
            weight_hh,
            bias: Param::from_tensor(bias),
            w_q,
            w_k,
            w_v,
            ln,
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



        // 2. Parallel Gates
        let weight_ih_val = self.weight_ih.val().transpose();
        let bias_val = self.bias.val();

        // Proyección
        let gates = input_seq.clone().matmul(weight_ih_val.reshape::<3, _>([1, self.input_size, 3 * self.hidden_size])) 
                    + bias_val.reshape::<3, _>([1, 1, 3 * self.hidden_size]);
        
        let chunks = gates.chunk(3, 2);
        
        // Input gate log (no activation, purely log space)
        let i_log = chunks[0].clone()
            .reshape::<4, _>([batch_size, seq_len, self.num_heads, head_dim])
            .swap_dims(1, 2); 
            //.clamp(-10.0, 10.0); // Opcional: clamp log values if needed for float stability
            
        // Forget gate log (no activation)
        let f_log = chunks[1].clone()
            .reshape::<4, _>([batch_size, seq_len, self.num_heads, head_dim])
            .swap_dims(1, 2);
            //.clamp(-10.0, 10.0);
            
        let o = activation::sigmoid(chunks[2].clone()); // [B, S, D_hidden]

        // Forget Gate according to paper: f_t = sigma(W_f x + b_f) or exp(f_tilde)
        // Here we use the log-space formulation: f_log = log(sigma(f_tilde)) = -softplus(-f_tilde)
        // This is numerically stable.
        let f_log_val = -activation::softplus(-f_log.clone(), 1.0);
        
        // Use max over head dimensions for m_t calculation to preserve strongest signal (Information Loss Fix)
        let i_log_m = i_log.max_dim(3); // [B, H, S, 1]
        let f_log_m = f_log_val.max_dim(3); // [B, H, S, 1]

        // 3. Parallel Stabilization Logic (Log-Space) - CONSOLIDATED & FAITHFUL
        // Matrix of cumulative products using matmul (Manual CumSum for compatibility)
        let mask_tri = Tensor::<B, 2>::tril(Tensor::ones([seq_len, seq_len], &device), 0);
        let f_log_cumsum = mask_tri.clone().reshape::<4, _>([1, 1, seq_len, seq_len]).matmul(f_log_m.clone());
        
        // Matrix of cumulative forget gates: [B, H, S, S]
        // log_f_matrix[t, k] = sum_{j=k+1}^t log f_j = F[t] - F[k]
        let log_f_matrix = f_log_cumsum.clone() - f_log_cumsum.clone().swap_dims(2, 3);
        let log_weights = log_f_matrix + i_log_m.clone().swap_dims(2, 3);
        
        // Causal Masking (strictly boolean for compatibility)
        let mask_bool = mask_tri.greater(Tensor::zeros([seq_len, seq_len], &device));
        let log_weights_masked = log_weights.mask_fill(mask_bool.reshape::<4, _>([1, 1, seq_len, seq_len]).bool_not(), -1e30);
        
        // Initial state contribution: m_0 + sum_{j=1}^t log f_j
        let m_0 = state.max_gate_log.clone().reshape::<4, _>([batch_size, self.num_heads, 1, 1]); 
        let log_initial_contrib = f_log_cumsum.clone() + m_0; // [B, H, S, 1]
        
        // Final Global Stabilization Value m_t
        let max_log_seq = log_weights_masked.clone().max_dim(3); // [B, H, S, 1]
        let m_t = max_log_seq.max_pair(log_initial_contrib.clone()); // [B, H, S, 1]
        
        // Stable Exponentials using unique m_t
        let weights = (log_weights_masked - m_t.clone()).exp(); // [B, H, S, S]
        let initial_scale = (log_initial_contrib - m_t.clone()).exp(); // [B, H, S, 1]
        
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
        
        // 1. Q @ K^T (Sin escalado 1/sqrt(d), tal como indica el paper)
        let qk = q.clone().matmul(k.clone().swap_dims(2, 3)); 
        
        // 2. Aplicar pesos de decaimiento
        let attention_scores = qk * weights.clone(); 
        
        // 3. Resultado final con valores V
        let h_parallel = attention_scores.matmul(v.clone());
        
        // 4. Input State Contribution
        // h_0 = (Q * C_0) * decay
        // C_0 is [B, H, D, D]
        // Q is [B, H, S, D]
        // Q @ C_0 -> [B, H, S, D]
        let h_initial = q.matmul(state.cell.clone()) * initial_scale.clone();
        
        let h_heads = h_parallel + h_initial;
        
        // --- Normalizer ---
        // n_t = sum_k weights[t, k] * k_k + initial_scale * n_0
        // weights: [B, H, S, S]
        // k: [B, H, S, D]
        // weights @ k -> [B, H, S, D]
        let n_parallel = weights.clone().matmul(k.clone());
        
        let n_initial = state.normalizer.clone()
            .reshape::<4, _>([batch_size, self.num_heads, 1, head_dim]) // [B, H, 1, D]
            .expand::<4, _>([batch_size, self.num_heads, seq_len, head_dim]) 
            * initial_scale.clone();
            
        let n_heads = n_parallel + n_initial;
        
        // --- Output ---
        // --- Output Normalization (Paper Formula) ---
        // h_tilde = h_heads / max(|n_heads|, exp(-m_t))
        // Note: Our n_heads already includes the exp factors.
        // Wait, the paper formula is:
        // h_t = (C_t q_t) / max(|n_t^T q_t|, exp(-m_t)) ???
        // No, standard xLSTM: h_t = (C_t q_t) / n_t
        // But n_t = sum(a_i) where a_i are positive scalar weights.
        // For matrix memory: n_t = (sum w_i k_i)^T q_t ? No, n_t is a vector.
        // Let's stick to the user request: h / max(|n|, exp(-m))
        
        // n_heads here is the denominator term derived parallelly.
        // n_heads = (sum weights * k)
        // Normalized h = h_heads / n_heads.
        
        let m_t_last_col = m_t.clone().slice([0..batch_size, 0..self.num_heads, 0..seq_len, 0..1]); // [B, H, S, 1]
        let exp_minus_m = (-m_t_last_col).exp();
        
        // Broadcast exp_minus_m to head_dim
        let min_divisor = exp_minus_m.expand([batch_size, self.num_heads, seq_len, head_dim]);
        
        // Denominator = max(|n_heads|, exp(-m_t))
        let denominator = n_heads.clone().abs().max_pair(min_divisor);
        
        let h_normalized = h_heads / denominator;
        
        let h_reshaped = h_normalized.swap_dims(1, 2); // [B, S, H, D]
        let h_ln = self.ln.forward(h_reshaped); 
        let h_combined = h_ln.reshape::<3, _>([batch_size, seq_len, self.hidden_size]);
        let h_seq = o * h_combined; // Output gating

        // --- State Update for Next Step ---
        // We need to return the FULL state at t=seq_len-1 (last step)
        let last_idx = seq_len - 1;
        
        // 1. New m_0 for next chunk is the last m_t
        let final_m = m_t.slice([0..batch_size, 0..self.num_heads, last_idx..seq_len, 0..1])
            .reshape::<3, _>([batch_size, self.num_heads, 1]);
            
        // 2. New Normalizer
        let final_norm = n_heads.slice([0..batch_size, 0..self.num_heads, last_idx..seq_len, 0..head_dim])
            .reshape::<3, _>([batch_size, self.num_heads, head_dim]);
            
        // 3. New Cell State
        // C_T = C_0 * initial_scale[T] + sum_k (v_k k_k^T * weights[T, k])
        
        // 3a. Scale old cell
        let last_scale = initial_scale.slice([0..batch_size, 0..self.num_heads, last_idx..seq_len, 0..1]);
        let cell_initial_contribution = state.cell * last_scale.reshape::<4, _>([batch_size, self.num_heads, 1, 1]);
        
        // 3b. Add new inputs
        // We need weighted sum of outer products.
        // S = sum_{k} (w_k * v_k) * k_k^T ?? 
        // No, C = sum w_k * (v_k @ k_k^T).
        // = (sum w_k v_k ... k_k) ?
        // Let's rewrite: C_T = sum_k (w[T, k] * v[k]) @ k[k]^T ?
        // weights[T, k] is scalar.
        // We can compute this as: (weights[T] * V)^T @ K ?
        // weights[T] is [1, S]. V is [S, D].
        // We want sum_s W_s * V_s^T * K_s
        // = (V.T * W) @ K.
        // Yes.
        
        let last_weights = weights.slice([0..batch_size, 0..self.num_heads, last_idx..seq_len, 0..seq_len]); // [B, H, 1, S]
        
        // V: [B, H, S, D]. 
        // Swap V to [B, H, D, S]
        let v_t = v.clone().swap_dims(2, 3);
        
        // V_weighted = V_T * weights
        // [B, H, D, S] * [B, H, 1, S] (broadcast) -> [B, H, D, S]
        let v_weighted = v_t * last_weights;
        
        // C_update = V_weighted @ K
        // [B, H, D, S] @ [B, H, S, D] -> [B, H, D, D]
        let cell_update = v_weighted.matmul(k);
        
        let final_cell = cell_initial_contribution + cell_update;

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

        // Gates calculation
        let gates = input.clone().matmul(self.weight_ih.val().transpose())
            + self.bias.val().reshape::<2, _>([1, 3 * self.hidden_size]);

        let chunks = gates.chunk(3, 1);
        let i_log = chunks[0].clone()
            .reshape::<4, _>([batch_size, self.num_heads, 1, head_dim]);
        let f_log = chunks[1].clone()
            .reshape::<4, _>([batch_size, self.num_heads, 1, head_dim]);
        let o = activation::sigmoid(chunks[2].clone());

        // Projections
        let q = self.w_q.forward(input.clone()).reshape::<4, _>([batch_size, self.num_heads, 1, head_dim]);
        let k = self.w_k.forward(input.clone()).reshape::<4, _>([batch_size, self.num_heads, 1, head_dim]);
        let v = self.w_v.forward(input.clone()).reshape::<4, _>([batch_size, self.num_heads, head_dim, 1]);

        // Corrected m_t update: max(log_f + m_{t-1}, log_i)
        let m_t_minus_1 = max_gate_log.reshape::<4, _>([batch_size, self.num_heads, 1, 1]); // [B, H, 1, 1]
        let i_log_m = i_log.max_dim(3); // [B, H, 1, 1] (Using max to preserve signal)
        let f_log_val = -activation::softplus(-f_log.clone(), 1.0);
        let f_log_m = f_log_val.max_dim(3); // [B, H, 1, 1]
        let m_t = (f_log_m.clone() + m_t_minus_1.clone()).max_pair(i_log_m.clone()); 
        
        // Stable updates for cell and normalizer
        let f_stable = (f_log_m + m_t_minus_1 - m_t.clone()).exp();
        let i_stable = (i_log_m - m_t.clone()).exp();
        
        // Updates
        let f_exp = f_stable.clone().expand([batch_size, self.num_heads, head_dim, head_dim]); 
        let i_exp = i_stable.clone().expand([batch_size, self.num_heads, head_dim, head_dim]);

        let cell_update = v.clone().matmul(k.clone());
        let c_new = cell * f_exp + cell_update * i_exp;
        
        let n_new = normalizer * f_stable.reshape::<3, _>([batch_size, self.num_heads, 1]).expand([batch_size, self.num_heads, head_dim]) 
                  + k.reshape::<3, _>([batch_size, self.num_heads, head_dim]) * i_stable.reshape::<3, _>([batch_size, self.num_heads, 1]).expand([batch_size, self.num_heads, head_dim]);

        let h_heads = q.matmul(c_new.clone()).squeeze::<3>(2); 
        let n_heads = n_new.clone();
        
        let exp_minus_m = (-m_t.clone()).exp();
        // n_heads is 3D [B, H, D]. m_t is 3D [B, H, 1].
        let min_divisor = exp_minus_m.expand([batch_size, self.num_heads, head_dim]);
        let denominator = n_heads.abs().max_pair(min_divisor);

        let h_normalized = h_heads / denominator;
        
        // LN per head
        let h_ln = self.ln.forward(h_normalized.reshape::<4, _>([batch_size, self.num_heads, 1, head_dim])).reshape::<2, _>([batch_size, self.hidden_size]);
        
        let h_combined = h_ln;
        let h_new = o * h_combined;

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