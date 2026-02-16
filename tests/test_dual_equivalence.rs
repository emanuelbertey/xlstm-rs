use xlstm::{MLstm, MLstmconfig, MLstmstate};
use burn::tensor::{Tensor, Distribution};

type TestBackend = burn_ndarray::NdArray<f32>;

fn main() {
    let device = Default::default();
    let batch_size = 2;
    let seq_len = 5;
    let input_size = 16;
    let hidden_size = 32;
    let num_heads = 4;
    
    let config = MLstmconfig::new(input_size, hidden_size, 1)
        .with_num_heads(num_heads)
        .with_dropout(0.0);
    
    let mlstm: MLstm<TestBackend> = config.init(&device);
    let cell = &mlstm.layers[0];
    
    let input_seq: Tensor<TestBackend, 3> = Tensor::random(
        [batch_size, seq_len, input_size], 
        Distribution::Default, 
        &device
    );
    
    let head_dim = hidden_size / num_heads;
    
    // Inicialización manual para evitar modificar src/mlstm.rs
    let initial_state = MLstmstate::new(
        Tensor::<TestBackend, 4>::zeros([batch_size, num_heads, head_dim, head_dim], &device),
        Tensor::<TestBackend, 2>::zeros([batch_size, hidden_size], &device),
        Tensor::<TestBackend, 3>::zeros([batch_size, num_heads, head_dim], &device),
        Tensor::<TestBackend, 3>::zeros([batch_size, num_heads, 1], &device),
    );
    
    // MODO PARALELO
    let (output_parallel, final_state_parallel) = cell.forward_sequence(&input_seq, initial_state.clone());
    
    // MODO RECURRENTE
    let mut current_state = initial_state;
    let mut outputs_recurrent = Vec::with_capacity(seq_len);
    
    for t in 0..seq_len {
        let input_t = input_seq.clone()
            .slice([0..batch_size, t, 0..input_size])
            .squeeze::<2>(1);
        
        let (output_t, new_state) = cell.forward(&input_t, current_state);
        outputs_recurrent.push(output_t.reshape::<3, _>([batch_size, 1, hidden_size]));
        current_state = new_state;
    }
    
    let output_recurrent = Tensor::<TestBackend, 3>::cat(outputs_recurrent, 1);
    
    // VERIFICACIÓN
    let diff = (output_parallel - output_recurrent).abs().mean().into_scalar();
    
    println!("Diferencia media: {:.2e}", diff);
    assert!(diff < 1e-5, "Diferencia demasiado alta: {:.2e}", diff);
    println!("✅ Test de equivalencia dual PASADO!");
}
