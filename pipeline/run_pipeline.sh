#!/usr/bin/env bash
# run_pipeline.sh
#
# Full post-fix pipeline: fix GLB eyes, validate, verify, render, copy deliverables.
# Run from the pipeline/ directory with Hee_Swim_fixed.glb present here.
#
# Usage:
#   cd pipeline
#   cp /path/to/Hee_Swim_fixed.glb .
#   cp /path/to/Hee_AR.usdz .          # (already done, for copying to output)
#   bash run_pipeline.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Step 1: Fix eyes in GLB ==="
node fix_eyes_glb.mjs

echo ""
echo "=== Step 2: Validate Hee_Swim_final.glb ==="
npx --yes gltf-transform validate Hee_Swim_final.glb

echo ""
echo "=== Step 3: Check extensions and file size ==="
node -e "
import('@gltf-transform/core').then(async ({ NodeIO }) => {
  const io = new NodeIO();
  const doc = await io.read('Hee_Swim_final.glb');
  const root = doc.getRoot();
  const exts = root.listExtensionsUsed().map(e => e.extensionName);
  const required = root.listExtensionsRequired().map(e => e.extensionName);
  console.log('Extensions used:', exts);
  console.log('Extensions required:', required);
  const missingMeshopt = !exts.includes('EXT_meshopt_compression');
  const missingWebP    = !exts.includes('EXT_texture_webp');
  console.log('meshopt absent:', missingMeshopt, '  (expected: true)');
  console.log('WebP absent:   ', missingWebP,    '  (expected: true)');
});
"
SIZE=$(wc -c < Hee_Swim_final.glb)
SIZE_MB=$(python3 -c "print(f'{$SIZE/1048576:.2f} MB')")
echo "File size: $SIZE_MB"
echo "(Expected: ~5 MB, similar to Hee_Swim_fixed.glb)"

echo ""
echo "=== Step 4: Skin verify ==="
node verify_eyes_glb.mjs

echo ""
echo "=== Step 5: Visual render ==="
python3 render_check_glb.py Hee_Swim_final.glb

echo ""
echo "=== Step 6: Copy deliverables ==="
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/user-data/outputs}"
mkdir -p "$OUTPUT_DIR" 2>/dev/null || OUTPUT_DIR="."
cp Hee_Swim_final.glb "$OUTPUT_DIR/Hee_Swim.glb"
echo "Copied Hee_Swim.glb → $OUTPUT_DIR"

if [ -f Hee_AR.usdz ]; then
  cp Hee_AR.usdz "$OUTPUT_DIR/Hee_AR.usdz"
  echo "Copied Hee_AR.usdz → $OUTPUT_DIR"
else
  echo "WARNING: Hee_AR.usdz not found — copy it here before running."
fi

echo ""
echo "All done. Deliverables in $OUTPUT_DIR:"
ls -lh "$OUTPUT_DIR"/*.glb "$OUTPUT_DIR"/*.usdz 2>/dev/null || ls -lh "$OUTPUT_DIR"
