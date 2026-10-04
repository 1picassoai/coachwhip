//! The route table the hot path learns from: counting steps and reading the logs back.
//! No GPU, no model file. Run with `cargo test --release` (debug builds of candle are slow).

// ---------------------------------------------------------------- experts.rs route table

#[test]
fn count_step_counts_every_pair_token_by_token() {
    let mut counts = coachwhip_engine::experts::Counts::new();
    let before = vec![vec![1, 2], vec![1]];
    let now = vec![vec![7], vec![7, 9]];
    coachwhip_engine::experts::count_step(&mut counts, 4, &before, &now);
    assert_eq!(counts[&(4, 1)][&7], 2);
    assert_eq!(counts[&(4, 1)][&9], 1);
    assert_eq!(counts[&(4, 2)][&7], 1);
    assert!(counts.get(&(4, 2)).map_or(true, |r| !r.contains_key(&9)));
    assert!(!counts.contains_key(&(5, 1)));
}

#[test]
fn count_step_ignores_a_token_count_mismatch_beyond_the_shorter_side() {
    let mut counts = coachwhip_engine::experts::Counts::new();
    coachwhip_engine::experts::count_step(&mut counts, 0, &[vec![1]], &[vec![2], vec![3]]);
    assert_eq!(counts[&(0, 1)][&2], 1);
    assert_eq!(counts.len(), 1);
}

#[test]
fn load_routes_reads_the_log_shape_and_skips_junk() {
    let dir = std::env::temp_dir().join(format!("coachwhip-routes-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // layers = 3: a "T" line then 3 layer lines; a stray header line and a short block are skipped.
    let log = "T -\n1 2,3\n4,5 6\n7,7\nT 1 2 3\n9\n9\nnot-a-number\nT -\n1\n2\n";
    std::fs::write(dir.join("a.log"), log).unwrap();
    std::fs::write(dir.join("ignored.txt"), "T -\n1\n2\n3\n").unwrap();
    let counts = coachwhip_engine::experts::load_routes(&[dir.as_path()], 3);
    std::fs::remove_dir_all(&dir).unwrap();
    // block 1, layer 0 -> 1: token0 (1,2)->(4), token1 (3)->(5,6)
    assert_eq!(counts[&(0, 1)][&4], 1);
    assert_eq!(counts[&(0, 2)][&4], 1);
    assert_eq!(counts[&(0, 3)][&5], 1);
    assert_eq!(counts[&(0, 3)][&6], 1);
    // block 1, layer 1 -> 2: (4)->(7), (5,6)->(7)
    assert_eq!(counts[&(1, 4)][&7], 1);
    assert_eq!(counts[&(1, 5)][&7], 1);
    // block 2: "not-a-number" parses to no experts, so (9)->(9) counted once and nothing from layer 2
    assert_eq!(counts[&(0, 9)][&9], 1);
    assert!(!counts.contains_key(&(1, 9)));
    // the last block is too short and must not be read
    assert!(!counts.contains_key(&(0, 1)) || counts[&(0, 1)].get(&2).is_none());
}

#[test]
fn load_routes_with_a_missing_folder_is_empty_not_an_error() {
    let counts = coachwhip_engine::experts::load_routes(&[std::path::Path::new("/nonexistent/coachwhip")], 4);
    assert!(counts.is_empty());
}
