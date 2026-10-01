//! Unit tests. No GPU, no model file: only the pure helpers and the CPU maths.
//! Run with `cargo test --release` (debug builds of candle are slow).

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};

// ---------------------------------------------------------------- chat.rs helpers

#[test]
fn html_escape_escapes_the_three_that_matter() {
    assert_eq!(crate::chat::html_escape("a&b<c>d"), "a&amp;b&lt;c&gt;d");
    assert_eq!(crate::chat::html_escape("plain"), "plain");
    assert_eq!(crate::chat::html_escape(""), "");
    // Escaping is applied once: an already-escaped amp is escaped again, as it must be.
    assert_eq!(crate::chat::html_escape("&amp;"), "&amp;amp;");
}

#[test]
fn markers_are_the_control_one_prefix_the_page_strips() {
    // The page's JS: MARK = '\u0001STATS:' and PROG = /\u0001PROG:(\{[^\n]*\})\n/
    assert_eq!(crate::chat::STATS_MARK, "\u{1}STATS:");
    assert_eq!(crate::chat::PROG_MARK, "\u{1}PROG:");
    assert!(crate::chat::PAGE.contains("const MARK = '\\u0001STATS:'"));
    assert!(crate::chat::PAGE.contains("const PROG = /\\u0001PROG:(\\{[^\\n]*\\})\\n/g"));
}

#[test]
fn progress_lines_match_the_page_regex() {
    // Same format strings as chat.rs builds in `handle`.
    let layer = format!("{}{{\"layer\":{},\"of\":{}}}\n", crate::chat::PROG_MARK, 3, 48);
    let summary = format!("{}{{\"summary\":{}}}\n", crate::chat::PROG_MARK, 16);
    for line in [&layer, &summary] {
        assert!(line.starts_with("\u{1}PROG:{"));
        assert!(line.ends_with("}\n"));
        let json = &line["\u{1}PROG:".len()..line.len() - 1];
        let v: serde_json::Value = serde_json::from_str(json).expect("progress line is JSON");
        assert!(v.is_object());
        assert!(!json.contains('\n'));
    }
}

#[test]
fn think_box_is_a_verbatim_substring_of_the_page() {
    // `PAGE.replace(THINK_BOX, "")` is a silent no-op if the two ever drift apart.
    assert!(crate::chat::PAGE.contains(crate::chat::THINK_BOX), "THINK_BOX no longer matches the page");
    assert_eq!(crate::chat::PAGE.matches(crate::chat::THINK_BOX).count(), 1);
    let without = crate::chat::PAGE.replace(crate::chat::THINK_BOX, "");
    assert!(!without.contains("id=\"think\""));
    // The JS still guards the missing element.
    assert!(without.contains("$('think')&&$('think').checked"));
}

#[test]
fn page_has_the_model_name_placeholder_once() {
    assert_eq!(crate::chat::PAGE.matches("MODEL_NAME").count(), 1);
}

/// Sends `bytes` through `chunk` over a loopback socket and returns what the other end saw.
fn framed(bytes: &[u8]) -> Vec<u8> {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let mut client = TcpStream::connect(addr).unwrap();
    let (mut server, _) = listener.accept().unwrap();
    crate::chat::chunk(&mut server, bytes).unwrap();
    drop(server);
    let mut got = Vec::new();
    client.read_to_end(&mut got).unwrap();
    got
}

#[test]
fn chunk_frames_as_hex_length_crlf_body_crlf() {
    assert_eq!(framed(b"hello"), b"5\r\nhello\r\n");
    assert_eq!(framed(&[b'x'; 255]), [b"ff\r\n".as_slice(), &[b'x'; 255], b"\r\n"].concat());
    assert_eq!(framed("é".as_bytes()), b"2\r\n\xc3\xa9\r\n");
}

#[test]
fn chunk_sends_nothing_for_empty_bytes() {
    // An empty chunk ("0\r\n\r\n") would end the HTTP response early.
    assert_eq!(framed(b""), b"");
}

#[test]
fn chunk_to_a_closed_socket_is_an_error_not_a_panic() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let client = TcpStream::connect(addr).unwrap();
    let (mut server, _) = listener.accept().unwrap();
    drop(client);
    // The first write may still succeed into the kernel buffer; keep writing until it fails.
    let mut failed = false;
    for _ in 0..64 {
        if crate::chat::chunk(&mut server, &[b'x'; 65536]).is_err() {
            failed = true;
            break;
        }
    }
    assert!(failed, "writing to a closed socket never reported an error");
}

// ---------------------------------------------------------------- experts.rs route table

#[test]
fn count_step_counts_every_pair_token_by_token() {
    let mut counts = crate::experts::Counts::new();
    let before = vec![vec![1, 2], vec![1]];
    let now = vec![vec![7], vec![7, 9]];
    crate::experts::count_step(&mut counts, 4, &before, &now);
    assert_eq!(counts[&(4, 1)][&7], 2);
    assert_eq!(counts[&(4, 1)][&9], 1);
    assert_eq!(counts[&(4, 2)][&7], 1);
    assert!(counts.get(&(4, 2)).map_or(true, |r| !r.contains_key(&9)));
    assert!(!counts.contains_key(&(5, 1)));
}

#[test]
fn count_step_ignores_a_token_count_mismatch_beyond_the_shorter_side() {
    let mut counts = crate::experts::Counts::new();
    crate::experts::count_step(&mut counts, 0, &[vec![1]], &[vec![2], vec![3]]);
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
    let counts = crate::experts::load_routes(&[dir.as_path()], 3);
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
    let counts = crate::experts::load_routes(&[std::path::Path::new("/nonexistent/coachwhip")], 4);
    assert!(counts.is_empty());
}

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
    let (o, s) = crate::model_next::delta_chunked(&q, &k, &v, &beta, &log_decay, s0).unwrap();
    assert_eq!(o.dims(), &[l, n_v, dv], "output shape for l = {l}");
    assert_eq!(s.dims(), &[n_v, dk, dv]);
    let od = max_abs_diff(&o, &o_ref);
    let sd = max_abs_diff(&s, &s_ref);
    assert!(od < 1e-4, "l = {l}: output differs from word-by-word by {od}");
    assert!(sd < 1e-4, "l = {l}: state differs from word-by-word by {sd}");
}

#[test]
fn delta_chunked_matches_word_by_word_when_l_is_a_whole_number_of_blocks() {
    let c = crate::model_next::DELTA_CHUNK;
    check_chunked_against_word_by_word(c, 1);
    check_chunked_against_word_by_word(2 * c, 2);
}

#[test]
fn delta_chunked_matches_word_by_word_with_padding() {
    let c = crate::model_next::DELTA_CHUNK;
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
    let (o_all, s_all) = crate::model_next::delta_chunked(&q, &k, &v, &beta, &log_decay, s0.clone()).unwrap();
    let cut = 45;
    let part = |t: &Tensor, from: usize, len: usize| t.narrow(0, from, len).unwrap().contiguous().unwrap();
    let (o1, s1) = crate::model_next::delta_chunked(&part(&q, 0, cut), &part(&k, 0, cut), &part(&v, 0, cut), &part(&beta, 0, cut), &part(&log_decay, 0, cut), s0).unwrap();
    let (o2, s2) = crate::model_next::delta_chunked(&part(&q, cut, l - cut), &part(&k, cut, l - cut), &part(&v, cut, l - cut), &part(&beta, cut, l - cut), &part(&log_decay, cut, l - cut), s1).unwrap();
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
    let (o, s) = crate::model_next::delta_chunked(&q, &k, &v, &beta, &log_decay, s0).unwrap();
    assert!(max_abs_diff(&o, &o_ref) < 1e-4);
    assert!(max_abs_diff(&s, &s_ref) < 1e-4);
}

// ---------------------------------------------------------------- loopback HTTP, no model

#[test]
fn chunk_output_is_valid_for_a_chunked_http_client() {
    // What the page receives per chunk: hex length, CRLF, body, CRLF. A zero-length chunk is never
    // produced by `chunk`; `handle` writes the terminator itself.
    let mut server_side: Vec<u8> = Vec::new();
    for piece in ["Here", "'s", " a", "\u{1}STATS:{\"written\":2}"] {
        server_side.extend(framed(piece.as_bytes()));
    }
    server_side.extend(b"0\r\n\r\n");
    // Decode by hand.
    let mut i = 0;
    let mut body = Vec::new();
    loop {
        let nl = server_side[i..].windows(2).position(|w| w == b"\r\n").unwrap() + i;
        let n = usize::from_str_radix(std::str::from_utf8(&server_side[i..nl]).unwrap(), 16).unwrap();
        if n == 0 {
            break;
        }
        body.extend_from_slice(&server_side[nl + 2..nl + 2 + n]);
        assert_eq!(&server_side[nl + 2 + n..nl + 4 + n], b"\r\n");
        i = nl + 4 + n;
    }
    let text = String::from_utf8(body).unwrap();
    let k = text.find(crate::chat::STATS_MARK).unwrap();
    assert_eq!(&text[..k], "Here's a");
    let stats: serde_json::Value = serde_json::from_str(&text[k + crate::chat::STATS_MARK.len()..]).unwrap();
    assert_eq!(stats["written"], 2);
}

#[allow(dead_code)]
fn _unused(_: &mut dyn Write) {}
