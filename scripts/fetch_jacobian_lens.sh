#!/usr/bin/env bash
# Clone the Jacobian-lens reference code at the commit our J-lens analysis used (its Qwen token glosses,
# assets/qwen_gloss.json.gz, label non-English readouts). Destination: third_party/jacobian-lens, or $JLENS_DIR.
set -euo pipefail
REPO=https://github.com/anthropics/jacobian-lens
COMMIT=581d398613e5602a5af361e1c34d3a92ea82ba8e
DEST=${JLENS_DIR:-$(cd "$(dirname "$0")/.." && pwd)/third_party/jacobian-lens}
[ -d "$DEST/.git" ] || git clone --quiet "$REPO" "$DEST"
git -C "$DEST" fetch --quiet origin "$COMMIT" 2>/dev/null || true
git -C "$DEST" checkout --quiet "$COMMIT"
echo "jacobian-lens at $(git -C "$DEST" rev-parse --short HEAD) in $DEST"
