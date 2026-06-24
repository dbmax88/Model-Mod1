#!/usr/bin/env python3
"""
render_check_glb.py

Quick visual verification that the eye repositioning worked.
Renders a top-down and side-view point cloud of Hee_Swim_final.glb
at rest pose using matplotlib, colouring body/eyes differently.

Requires: pip install pygltflib numpy matplotlib
"""

import sys
import struct
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

try:
    import pygltflib
except ImportError:
    print("Installing pygltflib …")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pygltflib", "--quiet"])
    import pygltflib


def get_accessor_data(gltf, accessor_idx):
    acc = gltf.accessors[accessor_idx]
    bv = gltf.bufferViews[acc.bufferView]
    buf_data = gltf.get_data_from_buffer_uri(gltf.buffers[bv.buffer].uri)
    if buf_data is None:
        # binary glb — use binary blob
        buf_data = bytes(gltf.binary_blob())

    component_types = {5120: 'b', 5121: 'B', 5122: 'h', 5123: 'H', 5125: 'I', 5126: 'f'}
    type_counts = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4,
                   'MAT2': 4, 'MAT3': 9, 'MAT4': 16}

    fmt = component_types[acc.componentType]
    n_comp = type_counts[acc.type]
    item_size = struct.calcsize(fmt) * n_comp
    byte_stride = bv.byteStride or item_size
    offset = (bv.byteOffset or 0) + (acc.byteOffset or 0)

    values = []
    for i in range(acc.count):
        chunk = buf_data[offset + i * byte_stride: offset + i * byte_stride + item_size]
        row = struct.unpack(f'<{n_comp}{fmt}', chunk)
        if acc.normalized:
            if acc.componentType == 5122:   # SIGNED_SHORT
                row = tuple(max(v / 32767.0, -1.0) for v in row)
            elif acc.componentType == 5120: # SIGNED_BYTE
                row = tuple(max(v / 127.0, -1.0) for v in row)
            elif acc.componentType == 5123: # UNSIGNED_SHORT
                row = tuple(v / 65535.0 for v in row)
            elif acc.componentType == 5121: # UNSIGNED_BYTE
                row = tuple(v / 255.0 for v in row)
        values.append(row)
    return np.array(values, dtype=np.float32)


def render_glb(path):
    gltf = pygltflib.GLTF2().load(path)

    mesh_positions = {}
    for mesh_idx, mesh in enumerate(gltf.meshes):
        prim = mesh.primitives[0]
        pos_acc_idx = prim.attributes.POSITION
        if pos_acc_idx is None:
            continue
        pos = get_accessor_data(gltf, pos_acc_idx)
        mesh_positions[mesh.name or f'mesh_{mesh_idx}'] = pos
        print(f"  {mesh.name}: {pos.shape[0]} vertices, X[{pos[:,0].min():.3f},{pos[:,0].max():.3f}] Y[{pos[:,1].min():.3f},{pos[:,1].max():.3f}] Z[{pos[:,2].min():.3f},{pos[:,2].max():.3f}]")

    if not mesh_positions:
        print("No meshes found!")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Eye fix verification — {Path(path).name}', fontsize=13)

    colors = {'body': '#1f77b4', 'eyes': '#ff7f0e', 'ground': '#aaaaaa'}
    default_colors = ['#2ca02c', '#d62728', '#9467bd']
    ci = 0

    for view_idx, (ax, (xlabel, ylabel, xcol, ycol, title)) in enumerate(zip(axes, [
        ('X', 'Y', 0, 1, 'Front view (X-Y)'),
        ('X', 'Z', 0, 2, 'Top view (X-Z)'),
    ])):
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        for name, pos in mesh_positions.items():
            key = next((k for k in colors if k in name.lower()), None)
            color = colors[key] if key else default_colors[ci % len(default_colors)]
            if key is None:
                ci += 1
            ax.scatter(pos[:, xcol], pos[:, ycol], s=0.5, c=color, alpha=0.6, label=name)

        ax.legend(markerscale=10, loc='upper right')

    plt.tight_layout()
    out = Path(path).with_suffix('.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print(f"Saved render to {out}")
    plt.close()


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'Hee_Swim_final.glb'
    print(f"Rendering {path} …")
    render_check_glb(path)
