//! The chat page and the HTTP framing: the pieces the browser depends on.
//! No GPU, no model file. Run with `cargo test --release` (debug builds of candle are slow).

use std::io::Read;
use std::net::{TcpListener, TcpStream};

// ---------------------------------------------------------------- chat.rs helpers

#[test]
fn html_escape_escapes_the_three_that_matter() {
    assert_eq!(coachwhip_server::chat::html_escape("a&b<c>d"), "a&amp;b&lt;c&gt;d");
    assert_eq!(coachwhip_server::chat::html_escape("plain"), "plain");
    assert_eq!(coachwhip_server::chat::html_escape(""), "");
    // Escaping is applied once: an already-escaped amp is escaped again, as it must be.
    assert_eq!(coachwhip_server::chat::html_escape("&amp;"), "&amp;amp;");
}

#[test]
fn markers_are_the_control_one_prefix_the_page_strips() {
    // The page's JS: MARK = '\u0001STATS:' and PROG = /\u0001PROG:(\{[^\n]*\})\n/
    assert_eq!(coachwhip_server::chat::STATS_MARK, "\u{1}STATS:");
    assert_eq!(coachwhip_server::chat::PROG_MARK, "\u{1}PROG:");
    assert!(coachwhip_server::chat::PAGE.contains("const MARK = '\\u0001STATS:'"));
    assert!(coachwhip_server::chat::PAGE.contains("const PROG = /\\u0001PROG:(\\{[^\\n]*\\})\\n/g"));
}

#[test]
fn progress_lines_match_the_page_regex() {
    // Same format strings as chat.rs builds in `handle`.
    let layer = format!("{}{{\"layer\":{},\"of\":{}}}\n", coachwhip_server::chat::PROG_MARK, 3, 48);
    let summary = format!("{}{{\"summary\":{}}}\n", coachwhip_server::chat::PROG_MARK, 16);
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
    assert!(coachwhip_server::chat::PAGE.contains(coachwhip_server::chat::THINK_BOX), "THINK_BOX no longer matches the page");
    assert_eq!(coachwhip_server::chat::PAGE.matches(coachwhip_server::chat::THINK_BOX).count(), 1);
    let without = coachwhip_server::chat::PAGE.replace(coachwhip_server::chat::THINK_BOX, "");
    assert!(!without.contains("id=\"think\""));
    // The JS still guards the missing element.
    assert!(without.contains("$('think')&&$('think').checked"));
}

#[test]
fn page_has_the_model_name_placeholder_once() {
    assert_eq!(coachwhip_server::chat::PAGE.matches("MODEL_NAME").count(), 1);
}

/// Sends `bytes` through `chunk` over a loopback socket and returns what the other end saw.
fn framed(bytes: &[u8]) -> Vec<u8> {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let mut client = TcpStream::connect(addr).unwrap();
    let (mut server, _) = listener.accept().unwrap();
    coachwhip_server::chat::chunk(&mut server, bytes).unwrap();
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
        if coachwhip_server::chat::chunk(&mut server, &[b'x'; 65536]).is_err() {
            failed = true;
            break;
        }
    }
    assert!(failed, "writing to a closed socket never reported an error");
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
    let k = text.find(coachwhip_server::chat::STATS_MARK).unwrap();
    assert_eq!(&text[..k], "Here's a");
    let stats: serde_json::Value = serde_json::from_str(&text[k + coachwhip_server::chat::STATS_MARK.len()..]).unwrap();
    assert_eq!(stats["written"], 2);
}
