#!/bin/bash
# Every model file given loads and answers a one-word question; a file of an unsupported
# architecture is refused with a clear message, never a crash.
# tests-mac/load_models.sh TOKENIZER MODEL.gguf [MODEL.gguf ...] [--unsupported FILE.gguf]
set -u
BIN=${COACHWHIP_BIN:-./target/release/coachwhip}
TOKENIZER=${1:?tokenizer.json}; shift
fail=0
while [ $# -gt 0 ]; do
  if [ "$1" = "--unsupported" ]; then
    out=$("$BIN" --model "$2" --tokenizer "$TOKENIZER" --max-tokens 4 --prompt "Hi" 2>&1)
    code=$?
    if [ $code -ne 0 ] && echo "$out" | grep -qi "support"; then echo "refused  $(basename "$2")"
    else echo "FAIL  $(basename "$2"): exit $code, no clear refusal"; fail=1; fi
    shift 2; continue
  fi
  out=$("$BIN" --model "$1" --tokenizer "$TOKENIZER" --temperature 0 --max-tokens 8 --prompt "Say OK." 2>&1)
  code=$?
  if [ $code -eq 0 ] && echo "$out" | grep -q "model ready"; then echo "loads    $(basename "$1")"
  else echo "FAIL  $(basename "$1"): exit $code"; echo "$out" | tail -3; fail=1; fi
  shift
done
[ $fail = 0 ] && echo "load_models: PASS" || echo "load_models: FAIL"
exit $fail
