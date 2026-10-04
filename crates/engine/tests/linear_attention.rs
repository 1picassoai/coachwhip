//! The 80B's linear attention: the block-at-a-time delta rule against the word-by-word one.
//! No GPU, no model file. Run with `cargo test --release` (debug builds of candle are slow).

// ---------------------------------------------------------------- model_next.rs delta rule (CPU)

use candle::{Device, Tensor};

/// A tiny deterministic generator so the test needs no `rand` crate.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }
    fn tensor(&mut self, shape: &[usize], scale: f32) -> Tensor {
        let n: usize = shape.iter().product();
        let v: Vec<f32> = (0..n).map(|_| self.next() * scale).collect();
        Tensor::from_vec(v, shape, &Device::Cpu).unwrap()
    }
}

/// The word-by-word delta rule exactly as model_next.rs steps it for one word (its `l == 1` path).
fn delta_word_by_word(q: &Tensor, k: &Tensor, v: &Tensor, beta: &Tensor, log_decay: &Tensor, mut s: Tensor) -> (Tensor, Tensor) {
    let (l, n_v, _dk) = q.dims3().unwrap();
    let decay = log_decay.exp().unwrap();
    let mut outs = Vec::with_capacity(l);
    for t in 0..l {
        let kt = k.get(t).unwrap().unsqueeze(1).unwrap().contiguous().unwrap();
        let qt = q.get(t).unwrap().unsqueeze(1).unwrap().contiguous().unwrap();
        let vt = v.get(t).unwrap().unsqueeze(1).unwrap();
        let bt = beta.get(t).unwrap().reshape((n_v, 1, 1)).unwrap();
        let gt = decay.get(t).unwrap().reshape((n_v, 1, 1)).unwrap();
        s = s.broadcast_mul(&gt).unwrap();
        let sk = kt.matmul(&s).unwrap();
        let d = (vt - sk).unwrap().broadcast_mul(&bt).unwrap();
        s = (s + kt.transpose(1, 2).unwrap().contiguous().unwrap().matmul(&d).unwrap()).unwrap();
        outs.push(qt.matmul(&s).unwrap().squeeze(1).unwrap());
    }
    (Tensor::stack(&outs, 0).unwrap(), s)
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    (a - b).unwrap().abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap()
}

fn inputs(seed: u64, l: usize, n_v: usize, dk: usize, dv: usize) -> (Tensor, Tensor, Tensor, Tensor, Tensor) {
    let mut g = Lcg(seed);
    let unit = |t: Tensor| -> Tensor {
        let n = t.sqr().unwrap().sum_keepdim(2).unwrap().sqrt().unwrap();
        t.broadcast_div(&n).unwrap()
    };
    let q = unit(g.tensor(&[l, n_v, dk], 1.0));
    let k = unit(g.tensor(&[l, n_v, dk], 1.0));
    let v = g.tensor(&[l, n_v, dv], 1.0);
    // beta in (0, 1); log decay in about (-0.6, 0): what sigmoid and softplus * A produce.
    let beta = candle_nn::ops::sigmoid(&g.tensor(&[l, n_v], 2.0)).unwrap();
    let log_decay = (g.tensor(&[l, n_v], 0.3).abs().unwrap() * -1.0).unwrap();
    (q, k, v, beta, log_decay)
}

fn check_chunked_against_word_by_word(l: usize, seed: u64) {
    let (n_v, dk, dv) = (3, 8, 6);
    let (q, k, v, beta, log_decay) = inputs(seed, l, n_v, dk, dv);
    let s0 = Tensor::zeros((n_v, dk, dv), candle::DType::F32, &Device::Cpu).unwrap();
    let (o_ref, s_ref) = delta_word_by_word(&q, &k, &v, &beta, &log_decay, s0.clone());
    let (o, s) = coachwhip_engine::model_next::delta_chunked(&q, &k, &v, &beta, &log_decay, s0).unwrap();
    assert_eq!(o.dims(), &[l, n_v, dv], "output shape for l = {l}");
    assert_eq!(s.dims(), &[n_v, dk, dv]);
    let od = max_abs_diff(&o, &o_ref);
    let sd = max_abs_diff(&s, &s_ref);
    assert!(od < 1e-4, "l = {l}: output differs from word-by-word by {od}");
    assert!(sd < 1e-4, "l = {l}: state differs from word-by-word by {sd}");
}

#[test]
fn delta_chunked_matches_word_by_word_when_l_is_a_whole_number_of_blocks() {
    let c = coachwhip_engine::model_next::DELTA_CHUNK;
    check_chunked_against_word_by_word(c, 1);
    check_chunked_against_word_by_word(2 * c, 2);
}

#[test]
fn delta_chunked_matches_word_by_word_with_padding() {
    let c = coachwhip_engine::model_next::DELTA_CHUNK;
    for l in [2, 5, c - 1, c + 1, 2 * c + 17] {
        check_chunked_against_word_by_word(l, 10 + l as u64);
    }
}

#[test]
fn delta_chunked_carries_state_across_calls() {
    // One call over 70 words must equal two calls, 70 = 45 + 25, with the state handed on.
    let (l, n_v, dk, dv) = (70, 3, 8, 6);
    let (q, k, v, beta, log_decay) = inputs(99, l, n_v, dk, dv);
    let s0 = Tensor::zeros((n_v, dk, dv), candle::DType::F32, &Device::Cpu).unwrap();
    let (o_all, s_all) = coachwhip_engine::model_next::delta_chunked(&q, &k, &v, &beta, &log_decay, s0.clone()).unwrap();
    let cut = 45;
    let part = |t: &Tensor, from: usize, len: usize| t.narrow(0, from, len).unwrap().contiguous().unwrap();
    let (o1, s1) = coachwhip_engine::model_next::delta_chunked(&part(&q, 0, cut), &part(&k, 0, cut), &part(&v, 0, cut), &part(&beta, 0, cut), &part(&log_decay, 0, cut), s0).unwrap();
    let (o2, s2) = coachwhip_engine::model_next::delta_chunked(&part(&q, cut, l - cut), &part(&k, cut, l - cut), &part(&v, cut, l - cut), &part(&beta, cut, l - cut), &part(&log_decay, cut, l - cut), s1).unwrap();
    let o_split = Tensor::cat(&[&o1, &o2], 0).unwrap();
    assert!(max_abs_diff(&o_all, &o_split) < 1e-4);
    assert!(max_abs_diff(&s_all, &s2) < 1e-4);
}

#[test]
fn delta_chunked_with_a_nonzero_starting_state() {
    let (l, n_v, dk, dv) = (37, 2, 8, 6);
    let (q, k, v, beta, log_decay) = inputs(7, l, n_v, dk, dv);
    let s0 = Lcg(123).tensor(&[n_v, dk, dv], 0.5);
    let (o_ref, s_ref) = delta_word_by_word(&q, &k, &v, &beta, &log_decay, s0.clone());
    let (o, s) = coachwhip_engine::model_next::delta_chunked(&q, &k, &v, &beta, &log_decay, s0).unwrap();
    assert!(max_abs_diff(&o, &o_ref) < 1e-4);
    assert!(max_abs_diff(&s, &s_ref) < 1e-4);
}
