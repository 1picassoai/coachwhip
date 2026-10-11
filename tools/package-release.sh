#!/bin/bash
# Builds the ready-built download for a release. Run it on an Apple silicon Mac from a clean
# checkout of the release tag; it writes dist/coachwhip-<version>-macos-arm64.tar.gz and its .sha256.
# Build paths are rewritten so no folder of the building Mac ends up inside the binaries.
set -euo pipefail
cd "$(dirname "$0")/.."
V="v$(grep -m1 '^version' Cargo.toml | cut -d'"' -f2)"
export RUSTFLAGS="--remap-path-prefix=$HOME/.cargo=/cargo --remap-path-prefix=$HOME/.rustup=/rustup --remap-path-prefix=$PWD=/coachwhip"
cargo build --release
name="coachwhip-$V-macos-arm64.tar.gz"
mkdir -p dist
COPYFILE_DISABLE=1 tar --no-mac-metadata --uid 0 --gid 0 --uname root --gname wheel -czf "dist/$name" \
  coachwhip.sh LICENSE target/release/coachwhip target/release/coachwhip-prepare
(cd dist && shasum -a 256 "$name" > "$name.sha256")
if strings target/release/coachwhip target/release/coachwhip-prepare | grep -q "$HOME"; then
  echo "error: a build path of this Mac is still inside the binaries" >&2
  exit 1
fi
echo "dist/$name"
cat "dist/$name.sha256"
