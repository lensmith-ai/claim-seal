#!/usr/bin/env bash
# render.sh: rebuild docs/readme/how-it-works.png from its source. Run from the repository root.
#   ANATOMY_KIT=<anatomy kit folder> PLAYWRIGHT_MODULES=<node_modules with playwright> BUILD_DIR=<scratch dir> \
#   bash docs/readme/src/render.sh
# The kit is https://github.com/wheresryan22/anatomy (MIT, Node only). Output lands in BUILD_DIR; look at it,
# then copy BUILD_DIR/how-it-works.png into docs/readme by hand.
set -euo pipefail
: "${ANATOMY_KIT:?set ANATOMY_KIT}" "${PLAYWRIGHT_MODULES:?set PLAYWRIGHT_MODULES}" "${BUILD_DIR:?set BUILD_DIR}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$BUILD_DIR"
rm -f "$BUILD_DIR/how-it-works.png" "$BUILD_DIR/how-it-works-card.html" "$BUILD_DIR/how-it-works.svg"
node "$SRC/build-how-it-works.mjs" "$BUILD_DIR"
node "$SRC/capture-still.mjs" "$BUILD_DIR/how-it-works-card.html" "$BUILD_DIR/how-it-works.png"
