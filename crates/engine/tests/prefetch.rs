//! The hot path's guess: scoring the next layer's experts from the route table.
//! No GPU, no model file.

use coachwhip_engine::experts::{count_step, next_scores, Counts};

fn table() -> Counts {
    let mut c = Counts::new();
    // Layer 3 -> 4: expert 1 led to 10 three times and to 11 once; expert 2 led to 11 twice.
    count_step(&mut c, 3, &[vec![1]], &[vec![10]]);
    count_step(&mut c, 3, &[vec![1]], &[vec![10]]);
    count_step(&mut c, 3, &[vec![1]], &[vec![10]]);
    count_step(&mut c, 3, &[vec![1]], &[vec![11]]);
    count_step(&mut c, 3, &[vec![2]], &[vec![11]]);
    count_step(&mut c, 3, &[vec![2]], &[vec![11]]);
    c
}

#[test]
fn a_pick_hands_its_weight_on_in_proportion_to_where_it_led() {
    let s = next_scores(&table(), 3, &[(1, 1.0)]);
    assert_eq!(s.len(), 2);
    assert_eq!(s[0].0, 10);
    assert!((s[0].1 - 0.75).abs() < 1e-6 && (s[1].1 - 0.25).abs() < 1e-6);
}

#[test]
fn several_picks_add_up() {
    let s = next_scores(&table(), 3, &[(1, 1.0), (2, 1.0)]);
    // 10: 0.75; 11: 0.25 + 1.0
    assert_eq!(s[0].0, 11);
    assert!((s[0].1 - 1.25).abs() < 1e-6);
    assert!((s[1].1 - 0.75).abs() < 1e-6);
}

#[test]
fn pick_weights_scale_their_share() {
    let s = next_scores(&table(), 3, &[(1, 0.5)]);
    assert!((s[0].1 - 0.375).abs() < 1e-6);
}

#[test]
fn unknown_experts_and_other_layers_add_nothing() {
    assert!(next_scores(&table(), 3, &[(99, 1.0)]).is_empty());
    assert!(next_scores(&table(), 4, &[(1, 1.0)]).is_empty());
}

#[test]
fn equal_scores_come_back_in_expert_order_every_time() {
    let mut c = Counts::new();
    for b in [30u32, 7, 19, 2] {
        count_step(&mut c, 0, &[vec![1]], &[vec![b]]);
    }
    for _ in 0..20 {
        let ids: Vec<u32> = next_scores(&c, 0, &[(1, 1.0)]).into_iter().map(|x| x.0).collect();
        assert_eq!(ids, vec![2, 7, 19, 30]);
    }
}
