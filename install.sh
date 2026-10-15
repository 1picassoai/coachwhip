#!/bin/bash
# Coachwhip installer: checks this Mac and downloads the ready-built Coachwhip for it. Bring your
# own model. Run it again at any time; it brings the install up to the release below.
#
#   curl -fsSL https://raw.githubusercontent.com/1picassoai/coachwhip/main/install.sh | bash
#
# COACHWHIP_DIR picks the install folder (default ~/coachwhip). COACHWHIP_VERSION picks the release
# tag (default: the latest signed release, below; it moves with every release).
# COACHWHIP_FROM_SOURCE=1 builds from source instead (installs Apple's command line tools and Rust).
set -euo pipefail

# Everything runs inside main, so a download cut off half way through runs nothing.
main() {
DIR="${COACHWHIP_DIR:-$HOME/coachwhip}"
REPO="${COACHWHIP_REPO:-https://github.com/1picassoai/coachwhip}"
VERSION="${COACHWHIP_VERSION:-v0.5.3}"
ASSET="coachwhip-$VERSION-macos-arm64.tar.gz"
DIST="${COACHWHIP_DIST:-$REPO/releases/download/$VERSION}"

say() { printf "\033[1;32m==>\033[0m %s\n" "$*"; }
die() { printf "\033[1;31merror:\033[0m %s\n" "$*" >&2; exit 1; }

# 1. This Mac
[ "$(uname -s)" = Darwin ] || die "Coachwhip runs on macOS only."
[ "$(uname -m)" = arm64 ] || die "Coachwhip needs an Apple silicon Mac (M1 or later)."
mem_gb=$(( $(sysctl -n hw.memsize) / 1073741824 ))
[ "$mem_gb" -ge 16 ] || die "Coachwhip needs 16 GB of memory; this Mac has $mem_gb GB."
need_gb=1
[ "${COACHWHIP_FROM_SOURCE:-0}" = 1 ] && need_gb=4
mkdir -p "$DIR"
free_gb=$(df -g "$DIR" | awk 'NR==2 {print $4}')
[ "$free_gb" -ge "$need_gb" ] || die "Coachwhip needs $need_gb GB of free disk; there are $free_gb GB."
say "Apple silicon, $mem_gb GB of memory, $free_gb GB free: good."

# 2. Coachwhip, ready-built
if [ "${COACHWHIP_FROM_SOURCE:-0}" != 1 ]; then
  say "Downloading Coachwhip $VERSION"
  tmp="$(mktemp -d)"
  if curl -fsSL -o "$tmp/$ASSET" "$DIST/$ASSET" && curl -fsSL -o "$tmp/$ASSET.sha256" "$DIST/$ASSET.sha256"; then
    (cd "$tmp" && shasum -a 256 -c "$ASSET.sha256" >/dev/null) || die "The download does not match its checksum; run the installer again."
    tar -xzf "$tmp/$ASSET" -C "$DIR"
    rm -rf "$tmp"
    xattr -dr com.apple.quarantine "$DIR" 2>/dev/null || true
    "$DIR/target/release/coachwhip" --help >/dev/null 2>&1 || die "Coachwhip downloaded but does not start on this Mac; try COACHWHIP_FROM_SOURCE=1."
    done_msg
    return
  fi
  rm -rf "$tmp"
  say "No ready-built Coachwhip for $VERSION; building it from source instead"
fi
from_source
done_msg
}

done_msg() {
say "Done. Start Coachwhip with your model and its tokenizer:"
printf "\n    %s/coachwhip.sh /path/to/model.gguf /path/to/tokenizer.json\n\nIt opens the chat in your browser. Stop it with Ctrl+C.\n\n" "$DIR"
}

# The source build, for COACHWHIP_FROM_SOURCE=1 or a release without a ready-built download.
from_source() {
# Apple's command line tools
if ! xcode-select -p >/dev/null 2>&1; then
  say "Installing Apple's command line tools. A window opens: finish it, then run this installer again."
  xcode-select --install || true
  exit 1
fi

# Rust
export PATH="$HOME/.cargo/bin:$PATH"
if ! command -v cargo >/dev/null 2>&1; then
  say "Installing Rust"
  curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal
fi

# The source
if [ -d "$DIR/.git" ]; then
  say "Updating Coachwhip to $VERSION"
  git -C "$DIR" fetch -q --depth 1 origin "refs/tags/$VERSION:refs/tags/$VERSION"
  git -C "$DIR" checkout -q "$VERSION"
elif [ -f "$DIR/Cargo.toml" ]; then
  say "Using the Coachwhip source already in $DIR"
else
  say "Downloading the Coachwhip $VERSION source"
  tmp="$(mktemp -d)"
  git clone -q --depth 1 --branch "$VERSION" "$REPO" "$tmp/coachwhip"
  cp -R "$tmp/coachwhip/." "$DIR/"
  rm -rf "$tmp"
fi
cd "$DIR"
say "Building Coachwhip (the first build takes about 10 minutes)"
if ! cargo build --release 2> build.log; then
  # Some Command Line Tools ship a beta SDK the linker cannot read; build against the one that
  # matches this macOS instead.
  major="$(sw_vers -productVersion | cut -d. -f1)"
  sdk="$(ls -d /Library/Developer/CommandLineTools/SDKs/MacOSX"$major".*.sdk 2>/dev/null | sort -t. -k2,2n | tail -1 || true)"
  [ -n "$sdk" ] || sdk="/Library/Developer/CommandLineTools/SDKs/MacOSX$major.sdk"
  if grep -q "unknown architecture" build.log && [ -d "$sdk" ]; then
    say "Building again against $(basename "$sdk")"
    SDKROOT="$sdk" cargo build --release 2> build.log || die "The build failed; see $DIR/build.log"
  else
    die "The build failed; see $DIR/build.log"
  fi
fi
}

main "$@"
