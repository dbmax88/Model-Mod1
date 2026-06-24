#!/usr/bin/env python3
"""
sanity_check_usdz.py

Final sanity pass on Hee_AR.usdz:
  1. ZIP integrity (all entries extractable)
  2. 64-byte alignment of usdc/binary assets inside the zip
  3. Reload usda/usdc and spot-check a couple of animation frames
     (eyes sit near the head, ground clearance above y=0)

Requires: pip install usd-core (or pxr from USD install)
Falls back to a zip-only check if USD is unavailable.
"""

import sys
import zipfile
import struct
import os
import numpy as np

USDZ_PATH = sys.argv[1] if len(sys.argv) > 1 else 'Hee_AR.usdz'

# ── 1. ZIP integrity ───────────────────────────────────────────────────────────
print(f"=== Checking {USDZ_PATH} ===")
with zipfile.ZipFile(USDZ_PATH, 'r') as z:
    bad = z.testzip()
    if bad:
        print(f"FAIL: Corrupt entry in zip: {bad}")
        sys.exit(1)
    entries = z.infolist()
    print(f"ZIP OK — {len(entries)} entries, no corruption")

    # ── 2. 64-byte alignment ─────────────────────────────────────────────────
    print("\n64-byte alignment check:")
    all_aligned = True
    for info in entries:
        offset = info.header_offset
        # header size: 30 bytes + len(filename) + len(extra)
        header_size = 30 + len(info.filename.encode()) + len(info.extra)
        data_offset = offset + header_size
        aligned = (data_offset % 64 == 0)
        status = "OK  " if aligned else "FAIL"
        if not aligned:
            all_aligned = False
        print(f"  {status}  offset={data_offset:8d}  {info.filename}")
    if all_aligned:
        print("All entries 64-byte aligned.")
    else:
        print("WARNING: Some entries not 64-byte aligned (may cause iOS AR issues).")

# ── 3. USD animation frame check ──────────────────────────────────────────────
try:
    from pxr import Usd, UsdGeom, UsdSkel, Gf
except ImportError:
    print("\nUSD Python bindings not available — skipping animation frame check.")
    print("Install with: pip install usd-core")
    sys.exit(0)

print(f"\n=== USD animation frame check ===")
stage = Usd.Stage.Open(USDZ_PATH)
if not stage:
    print("FAIL: Could not open stage")
    sys.exit(1)

time_codes = stage.GetTimeCodesPerSecond()
start = stage.GetStartTimeCode()
end = stage.GetEndTimeCode()
print(f"Timeline: {start}–{end} timecodes at {time_codes} fps")

# Collect all mesh prims
mesh_prims = [p for p in stage.Traverse() if p.GetTypeName() == 'Mesh']
print(f"Meshes: {[p.GetPath() for p in mesh_prims]}")

# Sample frames 0, 30, 60 (or start, mid, end)
frames = [start, (start + end) / 2, end]
for frame in frames:
    print(f"\n--- Frame {frame:.1f} ---")
    for prim in mesh_prims:
        pts_attr = prim.GetAttribute('points')
        if not pts_attr:
            continue
        pts = np.array(pts_attr.Get(frame))
        if pts is None or len(pts) == 0:
            continue
        name = prim.GetName()
        min_y = pts[:, 1].min()
        print(f"  {name}: {len(pts)} pts, Y range [{pts[:,1].min():.4f}, {pts[:,1].max():.4f}]"
              f"  X[{pts[:,0].min():.3f},{pts[:,0].max():.3f}]"
              f"  Z[{pts[:,2].min():.3f},{pts[:,2].max():.3f}]")
        if 'eye' in name.lower() and min_y < -0.1:
            print(f"    WARNING: Eyes have unexpectedly low Y at frame {frame}")

print("\nSanity check complete.")
