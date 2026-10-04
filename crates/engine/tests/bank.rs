//! The expert bank's bookkeeping: which expert sits in which slot, and who gets evicted.
//! No GPU, no model file: `Slots` is the bank without its memory.

use coachwhip_engine::experts::{Plan, Slots};

fn all_ready(_: usize) -> bool {
    true
}

/// The bank's rules that must hold after any sequence of forwards and prefetches.
fn check_consistent(s: &Slots, n: usize) {
    let mut seen: Vec<usize> = s.lru.iter().copied().collect();
    seen.sort_unstable();
    assert_eq!(seen, (0..n).collect::<Vec<_>>(), "the use order lists every slot exactly once");
    for (&e, &slot) in &s.slot_of {
        assert_eq!(s.holder[slot], Some(e), "slot_of and holder agree for expert {e}");
    }
    let held = s.holder.iter().filter(|h| h.is_some()).count();
    assert_eq!(held, s.slot_of.len(), "every held expert is findable");
    for e in &s.prefetched {
        assert!(s.slot_of.contains_key(e), "a prefetched expert is still in the bank");
    }
}

#[test]
fn a_fresh_bank_misses_everything_and_fills_slots_in_order() {
    let mut s = Slots::new(4);
    let plan = s.assign(&[7, 3], all_ready).unwrap();
    assert_eq!(plan, Plan { hits: vec![], misses: vec![(7, 0), (3, 1)], prefetch_hits: 0 });
    check_consistent(&s, 4);
}

#[test]
fn a_second_ask_is_a_hit_in_the_same_slot() {
    let mut s = Slots::new(4);
    s.assign(&[7, 3], all_ready).unwrap();
    let plan = s.assign(&[3], all_ready).unwrap();
    assert_eq!(plan.hits, vec![(3, 1)]);
    assert!(plan.misses.is_empty());
}

#[test]
fn the_least_recently_used_expert_is_evicted() {
    let mut s = Slots::new(2);
    s.assign(&[1], all_ready).unwrap();
    s.assign(&[2], all_ready).unwrap();
    s.assign(&[1], all_ready).unwrap(); // 1 is now the most recently used
    let plan = s.assign(&[3], all_ready).unwrap();
    assert_eq!(plan.misses, vec![(3, 1)], "3 takes 2's slot");
    assert!(!s.slot_of.contains_key(&2) && s.slot_of.contains_key(&1));
}

#[test]
fn never_evicts_an_expert_the_same_word_needs() {
    let mut s = Slots::new(3);
    s.assign(&[1, 2, 3], all_ready).unwrap(); // 1 is the least recently used
    let plan = s.assign(&[4, 1, 2], all_ready).unwrap();
    assert_eq!(plan.misses, vec![(4, 2)], "4 must take 3's slot, not 1's or 2's");
    assert_eq!(plan.hits.len(), 2);
    check_consistent(&s, 3);
}

#[test]
fn never_hands_out_a_slot_still_being_filled() {
    let mut s = Slots::new(3);
    let plan = s.assign(&[9], |slot| slot != 0).unwrap();
    assert_eq!(plan.misses, vec![(9, 1)], "slot 0 is being filled, so 9 goes to slot 1");
}

#[test]
fn a_crowded_bank_says_none_and_changes_nothing() {
    let mut s = Slots::new(2);
    s.assign(&[1, 2], all_ready).unwrap();
    let before = s.clone();
    assert!(s.assign(&[3, 4], |_| false).is_none(), "no slot is ready");
    assert!(s.assign(&[1, 3, 4], all_ready).is_none(), "three experts cannot fit two slots");
    assert_eq!(format!("{:?}", s.holder), format!("{:?}", before.holder));
    assert_eq!(s.lru, before.lru);
}

#[test]
fn reserve_skips_what_is_held_and_never_evicts_another_guess() {
    let mut s = Slots::new(3);
    s.assign(&[1, 2], all_ready).unwrap();
    let got = s.reserve(&[2, 5, 6], all_ready);
    assert_eq!(got.len(), 2, "2 is already held; 5 and 6 get slots");
    assert!(got.iter().all(|&(e, _)| e == 5 || e == 6));
    assert!(s.slot_of.contains_key(&2), "a guessed expert already in the bank is kept");
    assert!(s.prefetched.contains(&5) && s.prefetched.contains(&6));
    check_consistent(&s, 3);
}

#[test]
fn reserve_stops_when_no_slot_is_free() {
    let mut s = Slots::new(2);
    let got = s.reserve(&[1, 2, 3], all_ready);
    assert_eq!(got.len(), 2, "two slots, three guesses: the best two win");
    assert_eq!(got.iter().map(|g| g.0).collect::<Vec<_>>(), vec![1, 2]);
}

#[test]
fn a_used_guess_counts_as_a_prefetch_hit_once() {
    let mut s = Slots::new(4);
    s.reserve(&[8], all_ready);
    let plan = s.assign(&[8], all_ready).unwrap();
    assert_eq!(plan.prefetch_hits, 1);
    let again = s.assign(&[8], all_ready).unwrap();
    assert_eq!(again.prefetch_hits, 0, "the second use is an ordinary hit");
}

#[test]
fn an_unused_guess_that_is_evicted_is_forgotten() {
    let mut s = Slots::new(1);
    s.reserve(&[8], all_ready);
    s.assign(&[9], all_ready).unwrap();
    assert!(!s.prefetched.contains(&8));
    check_consistent(&s, 1);
}

/// A long random run, checked against a plain least-recently-used model after every step.
#[test]
fn matches_a_reference_lru_over_a_long_random_run() {
    let n = 8;
    let mut s = Slots::new(n);
    let mut reference: Vec<u32> = Vec::new(); // least recently used first
    let mut x: u64 = 42;
    let mut next = |m: u64| {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (x >> 33) % m
    };
    for _ in 0..5000 {
        let k = 1 + next(4) as usize;
        let mut needed: Vec<u32> = Vec::new();
        while needed.len() < k {
            let e = next(20) as u32;
            if !needed.contains(&e) {
                needed.push(e);
            }
        }
        let plan = s.assign(&needed, all_ready).unwrap();
        for &e in &needed {
            let was_held = reference.contains(&e);
            assert_eq!(plan.hits.iter().any(|h| h.0 == e), was_held, "hit or miss for {e}");
            if was_held {
                reference.retain(|&r| r != e);
            } else if reference.len() == n {
                let victim = *reference.iter().find(|r| !needed.contains(r)).unwrap();
                reference.retain(|&r| r != victim);
            }
            reference.push(e);
        }
        check_consistent(&s, n);
    }
}
