use xlstm::{MLstm, MLstmconfig, MLstmstate};
use burn::tensor::{Tensor, Distribution};
use burn::backend::Autodiff;

type TestBackend = burn_ndarray::NdArray<f32>;

fn run_equivalence() {
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
    let mut outputs_recurrent: Vec<Tensor<TestBackend, 3>> = Vec::with_capacity(seq_len);
    
    for t in 0..seq_len {
        let input_t: Tensor<TestBackend, 2> = input_seq.clone()
            .slice([0..batch_size, t..t+1, 0..input_size])
            .reshape([batch_size, input_size]);
        
        let (output_t, new_state) = cell.forward(&input_t, current_state);
        outputs_recurrent.push(output_t.reshape::<3, _>([batch_size, 1, hidden_size]));
        current_state = new_state;
    }
    
    let output_recurrent: Tensor<TestBackend, 3> = Tensor::cat(outputs_recurrent, 1);
    let final_state_recurrent = current_state;
    
    let output_diff = (output_parallel.clone() - output_recurrent.clone()).abs().mean().into_scalar();
    println!("Diferencia media en outputs: {:.2e}", output_diff);
    
    let cell_diff = (final_state_parallel.cell.clone() - final_state_recurrent.cell.clone()).abs().mean().into_scalar();
    println!("Diferencia media en cell states: {:.2e}", cell_diff);
    
    let norm_diff = (final_state_parallel.normalizer.clone() - final_state_recurrent.normalizer.clone()).abs().mean().into_scalar();
    println!("Diferencia media en normalizers: {:.2e}", norm_diff);
    
    if output_diff < 1e-4 {
        println!("✅ Test de equivalencia dual PASADO!");
    } else {
        println!("❌ ERROR: Los outputs no son equivalentes.");
        std::process::exit(1);
    }
}

fn run_grad_mlstm() {
    let device = Default::default();
    let batch_size = 1;
    let seq_len = 12;
    let input_size = 16;
    let hidden_size = 32;
    let num_heads = 4;

    type AdBackend = Autodiff<TestBackend>;

    let config = MLstmconfig::new(input_size, hidden_size, 1)
        .with_num_heads(num_heads)
        .with_dropout(0.0);
    
    let mlstm: MLstm<AdBackend> = config.init(&device);
    
    // Input con gradientes
    let x = Tensor::<AdBackend, 3>::random(
        [batch_size, seq_len, input_size], 
        Distribution::Normal(0.0, 1.0), 
        &device
    ).require_grad();

    // Forward a través de la secuencia paralela
    let (h_seq, _) = mlstm.forward(&x, None);
    
    // Tomamos el gradiente del último paso (cross-entropy dummy via sum)
    let h_last = h_seq.slice([0..batch_size, seq_len-1..seq_len, 0..hidden_size]).sum();
    
    let grads = h_last.backward();
    let x_grad = x.grad(&grads).expect("Debe existir gradiente para x en mLSTM");
    let grad_val = x_grad.abs().mean().into_scalar();

    println!("Gradiente REAL mLSTM (dual) |d last / d x|: {:.6}", grad_val);
    
    if grad_val > 1e-7 {
        println!("✅ Gradiente mLSTM saludable!");
    } else {
        println!("⚠️ ADVERTENCIA: Gradiente de mLSTM muy bajo ({:.2e})", grad_val);
    }
}

fn main() {
    println!("--- Ejecutando Equivalencia Dual/Serial ---");
    run_equivalence();
    println!("\n--- Ejecutando Test de Gradientes mLSTM ---");
    run_grad_mlstm();
}

