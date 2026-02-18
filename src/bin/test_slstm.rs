use xlstm::{SLstm, SLstmconfig, SLstmstate};
use burn::tensor::{Tensor, Distribution};
use burn::tensor::backend::Backend;
use burn::tensor::activation;
use burn::tensor::TensorData;
use rand::Rng;

type TestBackend = burn_ndarray::NdArray<f32>;

fn run_equivalence() {
    let device = Default::default();
    let batch_size = 2;
    let seq_len = 5;
    let input_size = 16;
    let hidden_size = 32;
    let config = SLstmconfig::new(input_size, hidden_size, 1).with_dropout(0.0);
    let slstm: SLstm<TestBackend> = config.init(&device);
    let layer = &slstm.layers[0];
    let input_seq: Tensor<TestBackend, 3> =
        Tensor::random([batch_size, seq_len, input_size], Distribution::Default, &device);
    let (output_parallel, final_states_parallel) = slstm.forward(&input_seq, None);
    let mut state = SLstmstate::new(
        Tensor::<TestBackend, 2>::zeros([batch_size, hidden_size], &device),
        Tensor::<TestBackend, 2>::zeros([batch_size, hidden_size], &device),
        Tensor::<TestBackend, 2>::zeros([batch_size, hidden_size], &device),
        Tensor::<TestBackend, 2>::zeros([batch_size, hidden_size], &device),
    );
    let projected_input = input_seq.clone().matmul(
        layer
            .weight_ih
            .val()
            .transpose()
            .unsqueeze_dim::<3>(0),
    ) + layer
        .bias
        .val()
        .unsqueeze_dim::<2>(0)
        .unsqueeze_dim::<3>(0);
    let mut outputs: Vec<Tensor<TestBackend, 3>> = Vec::with_capacity(seq_len);
    for t in 0..seq_len {
        let input_t = projected_input
            .clone()
            .slice([0..batch_size, t..(t + 1), 0..(4 * hidden_size)])
            .squeeze(1);
        let (h_new, new_state) = layer.forward_step(input_t, state);
        outputs.push(h_new.unsqueeze_dim(1));
        state = new_state;
    }
    let output_recurrent: Tensor<TestBackend, 3> = Tensor::cat(outputs, 1);
    let diff = (output_parallel.clone() - output_recurrent.clone()).abs().mean().into_scalar();
    println!("Diferencia media en outputs: {:.2e}", diff);
    let cell_diff = (final_states_parallel[0].cell.clone() - state.cell.clone()).abs().mean().into_scalar();
    println!("Diferencia media en cell states: {:.2e}", cell_diff);
    let norm_diff = (final_states_parallel[0].normalizer.clone() - state.normalizer.clone()).abs().mean().into_scalar();
    println!("Diferencia media en normalizers: {:.2e}", norm_diff);
}

fn run_stability() {
    let device = Default::default();
    let batch_size = 2;
    let seq_len = 20;
    let input_size = 16;
    let hidden_size = 32;
    let config = SLstmconfig::new(input_size, hidden_size, 1).with_dropout(0.0);
    let slstm: SLstm<TestBackend> = config.init(&device);
    let input = Tensor::<TestBackend, 3>::random([batch_size, seq_len, input_size], Distribution::Default, &device) * 100.0;
    let (out, states) = slstm.forward(&input, None);
    let out_mean = out.clone().abs().mean().into_scalar();
    let n_mean = states[0].normalizer.clone().abs().mean().into_scalar();
    let c_mean = states[0].cell.clone().abs().mean().into_scalar();
    println!("Estabilidad: |h|_mean={:.4}, |c|_mean={:.4}, |n|_mean={:.4}", out_mean, c_mean, n_mean);
}

fn run_monotonic() {
    let device = Default::default();
    let batch_size = 1;
    let seq_len = 20;
    let input_size = 1;
    let hidden_size = 8;
    let config = SLstmconfig::new(input_size, hidden_size, 1).with_dropout(0.0);
    let slstm: SLstm<TestBackend> = config.init(&device);
    let ones = Tensor::<TestBackend, 3>::ones([batch_size, seq_len, input_size], &device);
    let (out, _) = slstm.forward(&ones, None);
    let mut prev = 0.0f32;
    let mut non_decrease = 0usize;
    for t in 0..seq_len {
        let h_t = out.clone().slice([0..1, t..t+1, 0..hidden_size]).reshape([hidden_size]);
        let val = h_t.abs().mean().into_scalar();
        if val >= prev { non_decrease += 1; }
        prev = val;
    }
    println!("Monotonicidad aproximada en |h_t|: {}/{}", non_decrease, seq_len);
}

struct SimpleLstmCell<B: Backend> {
    w_ih: Tensor<B, 2>,
    w_hh: Tensor<B, 2>,
    b: Tensor<B, 1>,
}

impl<B: Backend> SimpleLstmCell<B> {
    fn new(input: usize, hidden: usize, device: &B::Device) -> Self {
        let w_ih = Tensor::<B, 2>::random([4 * hidden, input], Distribution::Default, device);
        let w_hh = Tensor::<B, 2>::random([4 * hidden, hidden], Distribution::Default, device);
        let b = Tensor::<B, 1>::zeros([4 * hidden], device);
        Self { w_ih, w_hh, b }
    }
    fn step(&self, x: Tensor<B, 2>, h: Tensor<B, 2>, c: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let gates = x.matmul(self.w_ih.clone().transpose()) + h.matmul(self.w_hh.clone().transpose()) + self.b.clone().unsqueeze_dim(0);
        let chunks = gates.chunk(4, 1);
        let i = activation::sigmoid(chunks[0].clone());
        let f = activation::sigmoid(chunks[1].clone());
        let g = chunks[2].clone().tanh();
        let o = activation::sigmoid(chunks[3].clone());
        let c_new = f * c + i * g;
        let h_new = o * c_new.clone().tanh();
        (h_new, c_new)
    }
}

fn run_compare_lstm() {
    let device = Default::default();
    let batch_size = 1;
    let seq_len = 20;
    let input_size = 4;
    let hidden_size = 16;
    let x = Tensor::<TestBackend, 3>::random([batch_size, seq_len, input_size], Distribution::Default, &device);
    let slstm_cfg = SLstmconfig::new(input_size, hidden_size, 1).with_dropout(0.0);
    let slstm: SLstm<TestBackend> = slstm_cfg.init(&device);
    let (h_seq, _) = slstm.forward(&x, None);
    let lstm_cell = SimpleLstmCell::<TestBackend>::new(input_size, hidden_size, &device);
    let mut h = Tensor::<TestBackend, 2>::zeros([batch_size, hidden_size], &device);
    let mut c = Tensor::<TestBackend, 2>::zeros([batch_size, hidden_size], &device);
    let mut hs = Vec::with_capacity(seq_len);
    for t in 0..seq_len {
        let x_t = x.clone().slice([0..batch_size, t..t+1, 0..input_size]).squeeze(1);
        let (h_new, c_new) = lstm_cell.step(x_t, h, c);
        h = h_new;
        c = c_new;
        hs.push(h.clone().unsqueeze_dim(1));
    }
    let h_seq_lstm = Tensor::<TestBackend, 3>::cat(hs, 1);
    let d = (h_seq.abs().mean().into_scalar(), h_seq_lstm.abs().mean().into_scalar());
    println!("Promedios |h| sLSTM vs LSTM: {:.4} vs {:.4}", d.0, d.1);
}

fn run_grad_input() {
    let device = Default::default();
    let batch_size = 1;
    let seq_len = 10;
    let input_size = 8;
    let hidden_size = 16;
    
    // Usamos Autodiff para medir el gradiente real
    type AtDiffBackend = burn::backend::Autodiff<TestBackend>;
    
    let config = SLstmconfig::new(input_size, hidden_size, 1).with_dropout(0.0);
    let slstm: SLstm<AtDiffBackend> = config.init(&device);
    
    let x = Tensor::<AtDiffBackend, 3>::random([batch_size, seq_len, input_size], Distribution::Normal(0.0, 1.0), &device)
        .require_grad();
        
    let (h_seq, _) = slstm.forward(&x, None);
    // Cambiamos a sum para ver la magnitud acumulada del gradiente
    let h_last = h_seq.slice([0..batch_size, seq_len-1..seq_len, 0..hidden_size]).sum();
    
    let grads = h_last.backward();
    let x_grad = x.grad(&grads).expect("Debe existir gradiente para x");
    let grad_val = x_grad.abs().mean().into_scalar();
    
    println!("Gradiente REAL medio |d last / d x|: {:.6}", grad_val);
}

fn run_copy_count() {
    let device = Default::default();
    let batch_size = 64;
    let seq_len = 12;
    let input_size = 1;
    let hidden_size = 16;
    let mut rng = rand::thread_rng(); // Usamos thread_rng compatible
    let mut xs: Vec<f32> = Vec::with_capacity(batch_size * seq_len);
    let mut ys: Vec<i64> = Vec::with_capacity(batch_size);
    for _ in 0..batch_size {
        let mut sum = 0i64;
        for _ in 0..seq_len {
            let bit: i64 = if rng.random_range(0..2) == 1 { 1 } else { 0 };
            xs.push(bit as f32);
            sum += bit;
        }
        ys.push(sum % 2);
    }
    let x = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(xs.clone(), [batch_size, seq_len, input_size]),
        &device,
    );
    let y = Tensor::<TestBackend, 1, burn::tensor::Int>::from_data(
        burn::tensor::TensorData::new(ys.clone(), [batch_size]),
        &device,
    );
    let config = SLstmconfig::new(input_size, hidden_size, 1).with_dropout(0.0).with_forget_bias(0.0);
    let slstm: SLstm<TestBackend> = config.init(&device);
    let w_s = Tensor::<TestBackend, 2>::random([hidden_size, 2], Distribution::Default, &device);
    let b_s = Tensor::<TestBackend, 1>::zeros([2], &device);
    let w_ih = Tensor::<TestBackend, 2>::random([4 * hidden_size, input_size], Distribution::Default, &device);
    let w_hh = Tensor::<TestBackend, 2>::random([4 * hidden_size, hidden_size], Distribution::Default, &device);
    let b = Tensor::<TestBackend, 1>::zeros([4 * hidden_size], &device);
    let w_l = Tensor::<TestBackend, 2>::random([hidden_size, 2], Distribution::Default, &device);
    let b_l = Tensor::<TestBackend, 1>::zeros([2], &device);
    let loss_fn = burn::nn::loss::CrossEntropyLossConfig::new().init(&device);
    for epoch in 0..5 {
        let (h_seq_s, _) = slstm.forward(&x.clone(), None);
        let h_last_s = h_seq_s.slice([0..batch_size, seq_len-1..seq_len, 0..hidden_size]).reshape([batch_size, hidden_size]);
        let logits_s = h_last_s.matmul(w_s.clone()) + b_s.clone().unsqueeze_dim(0);
        let loss_s = loss_fn.forward(logits_s, y.clone());
        let mut h = Tensor::<TestBackend, 2>::zeros([batch_size, hidden_size], &device);
        let mut c = Tensor::<TestBackend, 2>::zeros([batch_size, hidden_size], &device);
        for t in 0..seq_len {
            let x_t = x.clone().slice([0..batch_size, t..t+1, 0..input_size]).squeeze(1);
            let gates = x_t.matmul(w_ih.clone().transpose()) + h.matmul(w_hh.clone().transpose()) + b.clone().unsqueeze_dim(0);
            let chunks = gates.chunk(4, 1);
            let i = activation::sigmoid(chunks[0].clone());
            let f = activation::sigmoid(chunks[1].clone());
            let g = chunks[2].clone().tanh();
            let o = activation::sigmoid(chunks[3].clone());
            c = f * c + i * g;
            h = o * c.clone().tanh();
        }
        let logits_l = h.matmul(w_l.clone()) + b_l.clone().unsqueeze_dim(0);
        let loss_l = loss_fn.forward(logits_l, y.clone());
        let l_s = loss_s.clone().into_data().as_slice::<f32>().unwrap()[0];
        let l_l = loss_l.clone().into_data().as_slice::<f32>().unwrap()[0];
        if epoch % 1 == 0 || epoch == 4 {
            println!("copy_count epoch {}: sLSTM CE={:.4} LSTM CE={:.4}", epoch + 1, l_s, l_l);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() <= 1 {
        println!("Modo por defecto: ejecutar TODOS los tests de sLSTM");
        run_equivalence();
        run_stability();
        run_monotonic();
        run_compare_lstm();
        run_grad_input();
        run_copy_count();
        return;
    }
    let mode = args[1].as_str();
    match mode {
        "equiv" => run_equivalence(),
        "stability" => run_stability(),
        "monotonic" => run_monotonic(),
        "compare_lstm" => run_compare_lstm(),
        "grad" => run_grad_input(),
        "copy_count" => run_copy_count(),
        _ => {
            eprintln!("Modo inválido: {}", mode);
            eprintln!("Modos: equiv | stability | monotonic | compare_lstm | grad | copy_count");
            std::process::exit(1);
        }
    }
}
