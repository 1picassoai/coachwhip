//! The early guess: ranking a later layer's router scores, and dropping guesses that arrive too late.
//! No GPU, no model file.

use coachwhip_engine::experts::{is_stale, rank};

#[test]
fn rank_puts_the_best_scores_first() {
    assert_eq!(rank(&[0.1, 3.0, 2.0, -1.0, 1.0]), vec![1, 2, 4, 0, 3]);
}

#[test]
fn rank_breaks_ties_by_expert_id_so_a_guess_is_repeatable() {
    assert_eq!(rank(&[1.0, 1.0, 2.0, 1.0]), vec![2, 0, 1, 3]);
    for _ in 0..10 {
        assert_eq!(rank(&[0.5; 6]), vec![0, 1, 2, 3, 4, 5]);
    }
}

#[test]
fn rank_covers_every_expert_once() {
    let scores: Vec<f32> = (0..512).map(|i| ((i * 7919) % 97) as f32 * 0.01).collect();
    let mut ids = rank(&scores);
    assert_eq!(ids.len(), 512);
    assert!(ids.windows(2).all(|p| scores[p[0] as usize] >= scores[p[1] as usize]));
    ids.sort_unstable();
    assert_eq!(ids, (0..512).collect::<Vec<u32>>());
}

#[test]
fn rank_does_not_panic_on_nan() {
    let ids = rank(&[1.0, f32::NAN, 2.0]);
    assert_eq!(ids.len(), 3);
}

#[test]
fn a_guess_for_a_later_layer_of_the_same_word_is_fresh() {
    assert!(!is_stale(12, 7, 8, 7));
    assert!(!is_stale(9, 7, 8, 7));
}

#[test]
fn a_guess_is_stale_once_the_word_reaches_its_layer() {
    assert!(is_stale(8, 7, 8, 7), "the layer is reading its own misses now");
    assert!(is_stale(5, 7, 8, 7), "the word has passed that layer");
}

#[test]
fn a_guess_for_the_next_word_is_fresh_whatever_the_layer() {
    assert!(!is_stale(3, 8, 40, 7), "foresight reads for the coming word must survive");
}

#[test]
fn a_guess_from_an_earlier_word_is_stale_whatever_its_layer() {
    assert!(is_stale(40, 6, 3, 7));
    assert!(is_stale(0, 0, 0, 1));
}
