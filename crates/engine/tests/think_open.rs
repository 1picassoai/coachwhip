//! How a model's chat template opens the assistant's turn, read from the template text.
//! No GPU, no model file. Run with `cargo test --release`.

use coachwhip_engine::model::ThinkOpen;

const QWEN36: &str = r#"{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- else %}
        {{- '<think>\n' }}
    {%- endif %}
{%- endif %}"#;

const QWEN3: &str = r#"{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}"#;

const CODER: &str = r#"{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}"#;

#[test]
fn qwen36_opens_and_closes_thinking_from_the_prompt() {
    let t = ThinkOpen::from_template(Some(QWEN36));
    assert!(t.thinks());
    assert_eq!(t.off, "<think>\n\n</think>\n\n");
    assert_eq!(t.on, "<think>\n");
}

#[test]
fn qwen3_only_closes_thinking_from_the_prompt() {
    let t = ThinkOpen::from_template(Some(QWEN3));
    assert!(t.thinks());
    assert_eq!(t.off, "<think>\n\n</think>\n\n");
    assert_eq!(t.on, "");
}

#[test]
fn a_model_without_thinking_adds_nothing() {
    for template in [Some(CODER), None] {
        let t = ThinkOpen::from_template(template);
        assert!(!t.thinks());
        assert_eq!((t.off, t.on), ("", ""));
    }
}
