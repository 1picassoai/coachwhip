//! The router's choice: which experts each word goes to, and with what weight.
//! No GPU, no model file. Run with `cargo test --release` (debug builds of candle are slow).

use candle::{Device, Tensor};
use coachwhip_engine::experts::route;

fn logits(rows: &[&[f32]]) -> Tensor {
    let n = rows[0].len();
    let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    Tensor::from_vec(flat, (rows.len(), n), &Device::Cpu).unwrap()
}

fn softmax(row: &[f32]) -> Vec<f32> {
    let m = row.iter().cloned().fold(f32::MIN, f32::max);
    let e: Vec<f32> = row.iter().map(|x| (x - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.iter().map(|x| x / s).collect()
}

#[test]
fn picks_the_top_k_best_first() {
    let (ids, _) = route(&logits(&[&[0.1, 3.0, 2.0, -1.0, 1.0]]), 3, true).unwrap();
    assert_eq!(ids.to_vec2::<u32>().unwrap(), vec![vec![1, 2, 4]]);
}

#[test]
fn each_word_is_routed_on_its_own() {
    let (ids, _) = route(&logits(&[&[5.0, 0.0, 1.0, 2.0], &[0.0, 1.0, 9.0, 3.0]]), 2, true).unwrap();
    assert_eq!(ids.to_vec2::<u32>().unwrap(), vec![vec![0, 3], vec![2, 3]]);
}

#[test]
fn renormalised_weights_sum_to_one_and_keep_their_ratios() {
    let row = [0.5f32, 2.0, 1.0, -0.5, 1.5];
    let (_, w) = route(&logits(&[&row]), 3, true).unwrap();
    let w = &w.to_vec2::<f32>().unwrap()[0];
    assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    let p = softmax(&row);
    let kept = p[1] + p[4] + p[2];
    for (got, i) in w.iter().zip([1usize, 4, 2]) {
        assert!((got - p[i] / kept).abs() < 1e-5, "weight of expert {i}");
    }
}

#[test]
fn without_renormalising_the_weights_are_the_router_probabilities() {
    let row = [1.0f32, 0.0, 3.0, 2.0];
    let (_, w) = route(&logits(&[&row]), 2, false).unwrap();
    let w = &w.to_vec2::<f32>().unwrap()[0];
    let p = softmax(&row);
    assert!((w[0] - p[2]).abs() < 1e-5 && (w[1] - p[3]).abs() < 1e-5);
    assert!(w.iter().sum::<f32>() < 1.0);
}

#[test]
fn weights_follow_the_order_of_the_picks() {
    let (_, w) = route(&logits(&[&[0.0, 4.0, 1.0, 2.0, 3.0]]), 4, true).unwrap();
    let w = &w.to_vec2::<f32>().unwrap()[0];
    assert!(w.windows(2).all(|p| p[0] >= p[1]));
}
