//! The OpenAI door: who may call it, how long an answer may be, and the chat format the model sees.
//! No GPU, no model file.

use coachwhip_server::chat::{answer_budget, chat_prompt, message_text, own_page, CONTEXT_LIMIT};
use serde_json::json;

const PORT: u16 = 8090;

#[test]
fn this_machine_may_call() {
    assert!(own_page("127.0.0.1:8090", "", PORT));
    assert!(own_page("localhost:8090", "", PORT));
    assert!(own_page("127.0.0.1:8090", "http://127.0.0.1:8090", PORT));
    assert!(own_page("localhost:8090", "http://localhost:8090", PORT));
}

#[test]
fn another_site_in_the_browser_may_not() {
    assert!(!own_page("127.0.0.1:8090", "http://evil.example", PORT));
    assert!(!own_page("127.0.0.1:8090", "https://127.0.0.1:8090", PORT), "only the page's own http origin");
    assert!(!own_page("127.0.0.1:8090", "null", PORT));
    assert!(!own_page("127.0.0.1:8090", "http://127.0.0.1:9999", PORT));
}

#[test]
fn a_rebound_or_wrong_host_may_not() {
    assert!(!own_page("evil.example:8090", "", PORT));
    assert!(!own_page("127.0.0.1:9999", "", PORT));
    assert!(!own_page("127.0.0.1:8090.evil.example", "", PORT));
    assert!(!own_page("", "", PORT));
}

#[test]
fn no_limit_asked_gets_the_room_left() {
    assert_eq!(answer_budget(None, 4000, 100), 4000);
    assert_eq!(answer_budget(None, 4000, CONTEXT_LIMIT - 300), 300);
    assert_eq!(answer_budget(None, 4000, CONTEXT_LIMIT + 5), 0);
}

#[test]
fn an_asked_limit_is_kept_but_never_above_coachwhips() {
    assert_eq!(answer_budget(Some(200), 4000, 100), 200);
    assert_eq!(answer_budget(Some(10_000), 4000, 100), 4000);
}

#[test]
fn the_context_limit_is_16k() {
    assert_eq!(CONTEXT_LIMIT, 16384);
}

#[test]
fn messages_become_the_models_chat_format() {
    let p = chat_prompt(&[json!({"role": "system", "content": "Be brief."}), json!({"role": "user", "content": "Hi"})], "");
    assert_eq!(p, "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n");
}

#[test]
fn a_thinking_models_turn_opens_the_way_its_template_says() {
    let p = chat_prompt(&[json!({"role": "user", "content": "Hi"})], "<think>\n\n</think>\n\n");
    assert!(p.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
}

#[test]
fn unknown_roles_are_mapped_safely() {
    let p = chat_prompt(&[json!({"role": "developer", "content": "a"}), json!({"role": "tool", "content": "b"}), json!({"content": "c"})], "");
    assert!(p.starts_with("<|im_start|>system\na<|im_end|>\n<|im_start|>user\nb<|im_end|>\n<|im_start|>user\nc<|im_end|>\n"));
    assert!(p.ends_with("<|im_start|>assistant\n"));
}

#[test]
fn content_parts_keep_only_their_text() {
    assert_eq!(message_text(&json!("plain")), "plain");
    assert_eq!(message_text(&json!([{"type": "text", "text": "one"}, {"type": "image_url", "image_url": {"url": "x"}}, {"type": "text", "text": "two"}])), "one\ntwo");
    assert_eq!(message_text(&json!(null)), "");
    assert_eq!(message_text(&json!(42)), "");
}
