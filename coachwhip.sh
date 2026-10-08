#!/bin/bash
# Start the Coachwhip chat and open it in the browser.
#   ./coachwhip.sh <model.gguf> <tokenizer.json> [bank] [port] [more coachwhip options...]
set -euo pipefail
[ $# -ge 2 ] || { echo "usage: $0 <model.gguf> <tokenizer.json> [bank] [port]" >&2; exit 1; }
# Paths are made absolute before moving to the Coachwhip folder, so relative ones work too.
abs() { [ -f "$1" ] || { echo "no such file: $1" >&2; exit 1; }; echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; }
MODEL="$(abs "$1")"
TOKENIZER="$(abs "$2")"
BANK="${3:-44}"
PORT="${4:-8090}"
cd "$(dirname "$0")"
./target/release/coachwhip --model "$MODEL" --tokenizer "$TOKENIZER" --bank "$BANK" --chat "$PORT" "${@:5}" &
pid=$!
trap 'kill $pid 2>/dev/null' INT TERM EXIT
until curl -s -o /dev/null "http://127.0.0.1:$PORT/"; do
  kill -0 $pid 2>/dev/null || exit 1
  sleep 1
done
open "http://127.0.0.1:$PORT/"
wait $pid
