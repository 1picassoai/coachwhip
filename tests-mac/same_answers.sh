#!/bin/bash
# Speed settings must never change what the model says. Runs one prompt at temperature 0 with
# different banks and prefetch settings and checks every answer is byte-identical.
# Needs a Mac with the model: tests-mac/same_answers.sh /path/to/model.gguf /path/to/tokenizer.json
set -u
MODEL=${1:?model.gguf}
TOKENIZER=${2:?tokenizer.json}
BIN=${COACHWHIP_BIN:-./target/release/coachwhip}
PROMPT="Write a Python function that merges overlapping intervals, with a short explanation."
OUT=$(mktemp -d)
fail=0

run() {
  local tag=$1; shift
  "$BIN" --model "$MODEL" --tokenizer "$TOKENIZER" --temperature 0 --max-tokens 120 --prompt "$PROMPT" "$@" > "$OUT/$tag.txt" 2> "$OUT/$tag.log" || { echo "FAIL $tag: exit $?"; tail -3 "$OUT/$tag.log"; fail=1; }
}

run bank44 --bank 44
run bank56 --bank 56
run no-prefetch --bank 56 --prefetch 0
run prefetch8 --bank 56 --prefetch 8

ref=$(shasum -a 256 < "$OUT/bank44.txt" | cut -c1-16)
for t in bank56 no-prefetch prefetch8; do
  got=$(shasum -a 256 < "$OUT/$t.txt" | cut -c1-16)
  if [ "$got" = "$ref" ]; then echo "same  $t"; else echo "DIFF  $t (see $OUT)"; fail=1; fi
done
[ -s "$OUT/bank44.txt" ] || { echo "FAIL: empty answer"; fail=1; }
[ $fail = 0 ] && { echo "same_answers: PASS"; rm -rf "$OUT"; } || echo "same_answers: FAIL (files kept in $OUT)"
exit $fail
