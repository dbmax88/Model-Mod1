#!/usr/bin/env python3
"""
GLB → animated USDZ conversion for giant_pacific_octopus_swim_A3.glb
Produces giant_pacific_octopus_swim_A3.usdz with UsdSkel animation.
"""

import struct, json, math, os, sys, zipfile, shutil, re
import numpy as np
from pathlib import Path

from pxr import (Usd, UsdGeom, UsdSkel, UsdShade, Sdf, Gf, Vt, UsdUtils,
                 UsdLux, Kind)

GLB_IN  = "giant_pacific_octopus_swim_A3.glb"
WORK_DIR = Path("usd_work");  WORK_DIR.mkdir(exist_ok=True)
USDA_OUT = str(WORK_DIR / "octopus.usda")
USDC_OUT = str(WORK_DIR / "octopus.usdc")
USDZ_OUT = "giant_pacific_octopus_swim_A3.usdz"

FPS = 24.0
N_FRAMES = 101  # 0..100

# ─── GLB reading ─────────────────────────────────────────────────────────────

def read_glb(path):
    with open(path, "rb") as f:
        magic, ver, _ = struct.unpack("<III", f.read(12))
        assert magic == 0x46546C67 and ver == 2, "Not a valid GLB"
        chunks = {}
        while True:
            h = f.read(8)
            if len(h) < 8: break
            clen, ctype = struct.unpack("<II", h)
            chunks[ctype] = f.read(clen)
    gltf = json.loads(chunks[0x4E4F534A])
    return gltf, chunks.get(0x004E4942, b"")

def acc_data(gltf, bin_data, idx, flat=False):
    """Return accessor as numpy array, shape (count, components)."""
    if idx is None: return None
    acc = gltf["accessors"][idx]
    bv  = gltf["bufferViews"][acc["bufferView"]]
    ct  = acc["componentType"]
    tp  = acc["type"]
    cnt = acc["count"]
    dtype_map = {5120:np.int8, 5121:np.uint8, 5122:np.int16,
                 5123:np.uint16, 5125:np.uint32, 5126:np.float32}
    type_nc   = {"SCALAR":1,"VEC2":2,"VEC3":3,"VEC4":4,
                 "MAT2":4,"MAT3":9,"MAT4":16}
    dt = dtype_map[ct]; nc = type_nc[tp]
    stride = bv.get("byteStride") or (np.dtype(dt).itemsize * nc)
    offset = bv.get("byteOffset",0) + acc.get("byteOffset",0)
    raw = np.frombuffer(bin_data, dtype=dt,
                        count=cnt*nc,
                        offset=offset) if stride == np.dtype(dt).itemsize*nc \
          else _strided_read(bin_data, offset, cnt, nc, dt, stride)
    arr = raw.reshape(cnt, nc).astype(np.float64)
    if acc.get("normalized"):
        if ct == 5122: arr = np.clip(arr / 32767.0, -1, 1)
        elif ct == 5121: arr = arr / 255.0
        elif ct == 5123: arr = arr / 65535.0
        elif ct == 5120: arr = np.clip(arr / 127.0, -1, 1)
    return arr.flatten() if flat else arr

def _strided_read(data, offset, cnt, nc, dt, stride):
    itemsz = np.dtype(dt).itemsize
    rows = []
    for i in range(cnt):
        row = np.frombuffer(data, dtype=dt, count=nc,
                            offset=offset + i*stride)
        rows.append(row)
    return np.array(rows)

def acc_int(gltf, bin_data, idx):
    """Accessor as integer array (for indices/joints)."""
    acc = gltf["accessors"][idx]
    bv  = gltf["bufferViews"][acc["bufferView"]]
    ct  = acc["componentType"]
    cnt = acc["count"]
    tp  = acc["type"]
    dtype_map = {5120:np.int8, 5121:np.uint8, 5122:np.int16,
                 5123:np.uint16, 5125:np.uint32, 5126:np.float32}
    type_nc   = {"SCALAR":1,"VEC2":2,"VEC3":3,"VEC4":4}
    dt = dtype_map[ct]; nc = type_nc[tp]
    stride = bv.get("byteStride") or (np.dtype(dt).itemsize * nc)
    offset = bv.get("byteOffset",0) + acc.get("byteOffset",0)
    raw = np.frombuffer(bin_data, dtype=dt, count=cnt*nc, offset=offset)
    return raw.reshape(cnt, nc).astype(np.int32)

# ─── Node hierarchy ──────────────────────────────────────────────────────────

def build_parents(gltf):
    """parent[i] = parent node index, or None for roots."""
    nodes = gltf["nodes"]
    parent = [None] * len(nodes)
    for i, nd in enumerate(nodes):
        for ch in nd.get("children", []):
            parent[ch] = i
    return parent

def node_local_mat(nd):
    """4x4 local matrix from node TRS or matrix (row-major numpy)."""
    if "matrix" in nd:
        return np.array(nd["matrix"], dtype=np.float64).reshape(4,4,order='F')
    t = np.array(nd.get("translation",[0,0,0]), dtype=np.float64)
    r = np.array(nd.get("rotation",[0,0,0,1]), dtype=np.float64)  # xyzw
    s = np.array(nd.get("scale",[1,1,1]), dtype=np.float64)
    return trs_to_mat(t, r, s)

def trs_to_mat(t, r, s):
    """Build 4x4 from translation, quaternion(xyzw), scale."""
    x,y,z,w = r
    sx,sy,sz = s
    m = np.array([
        [(1-2*(y*y+z*z))*sx, (2*(x*y+z*w))*sx,   (2*(x*z-y*w))*sx,   0],
        [(2*(x*y-z*w))*sy,   (1-2*(x*x+z*z))*sy,  (2*(y*z+x*w))*sy,   0],
        [(2*(x*z+y*w))*sz,   (2*(y*z-x*w))*sz,    (1-2*(x*x+y*y))*sz, 0],
        [t[0], t[1], t[2], 1]
    ], dtype=np.float64)
    return m

def compute_world_mats(gltf, parent):
    """Compute world matrix for every node."""
    nodes = gltf["nodes"]
    world = [None] * len(nodes)
    def get(i):
        if world[i] is None:
            lm = node_local_mat(nodes[i])
            p = parent[i]
            world[i] = get(p) @ lm if p is not None else lm
        return world[i]
    for i in range(len(nodes)): get(i)
    return world

def mat_to_trs(m):
    """Decompose 4x4 (row-major) into translation, quaternion(xyzw), scale."""
    t = m[3, :3].copy()
    sx = np.linalg.norm(m[0,:3])
    sy = np.linalg.norm(m[1,:3])
    sz = np.linalg.norm(m[2,:3])
    r = m[:3,:3] / np.array([sx,sy,sz])  # remove scale
    # rotation matrix → quaternion
    trace = r[0,0]+r[1,1]+r[2,2]
    if trace > 0:
        s = 0.5/math.sqrt(trace+1)
        w = 0.25/s
        x = (r[1,2]-r[2,1])*s; y = (r[2,0]-r[0,2])*s; z = (r[0,1]-r[1,0])*s
    elif r[0,0]>r[1,1] and r[0,0]>r[2,2]:
        s = 2*math.sqrt(1+r[0,0]-r[1,1]-r[2,2])
        w=(r[1,2]-r[2,1])/s; x=0.25*s; y=(r[0,1]+r[1,0])/s; z=(r[0,2]+r[2,0])/s
    elif r[1,1]>r[2,2]:
        s = 2*math.sqrt(1+r[1,1]-r[0,0]-r[2,2])
        w=(r[2,0]-r[0,2])/s; x=(r[0,1]+r[1,0])/s; y=0.25*s; z=(r[1,2]+r[2,1])/s
    else:
        s = 2*math.sqrt(1+r[2,2]-r[0,0]-r[1,1])
        w=(r[0,1]-r[1,0])/s; x=(r[0,2]+r[2,0])/s; y=(r[1,2]+r[2,1])/s; z=0.25*s
    q = np.array([x,y,z,w], dtype=np.float64)
    q /= np.linalg.norm(q)
    return t, q, np.array([sx,sy,sz])

# ─── Joint topology ──────────────────────────────────────────────────────────

def topo_sort_joints(joint_node_indices, parent):
    """Return joint_node_indices in parent-before-child order."""
    joint_set = set(joint_node_indices)
    # Build parent within the joint set
    joint_parent = {}
    for j in joint_node_indices:
        p = parent[j]
        while p is not None and p not in joint_set:
            p = parent[p]
        joint_parent[j] = p  # None if root joint
    # BFS from roots
    roots = [j for j in joint_node_indices if joint_parent[j] is None]
    children = {j: [] for j in joint_node_indices}
    for j in joint_node_indices:
        p = joint_parent[j]
        if p is not None:
            children[p].append(j)
    order = []
    queue = list(roots)
    while queue:
        j = queue.pop(0)
        order.append(j)
        queue.extend(children[j])
    assert len(order) == len(joint_node_indices), \
        f"Topo sort lost joints: {len(order)} vs {len(joint_node_indices)}"
    return order, joint_parent

def _topo_sort_nodes(node_set, parent_map):
    """Sort nodes in node_set parent-before-child (DFS pre-order)."""
    order = []
    visited = set()
    def visit(n):
        if n in visited: return
        visited.add(n)
        p = parent_map[n]
        if p is not None and p in node_set:
            visit(p)
        order.append(n)
    for n in sorted(node_set):
        visit(n)
    return order

def joint_usd_paths(sorted_joints, joint_parent, gltf):
    """Build USD joint path strings (parent/child/...) for sorted joints."""
    nodes = gltf["nodes"]
    # sanitize name for USD path component
    def san(s):
        s = re.sub(r'[^A-Za-z0-9_]', '_', s)
        if s and s[0].isdigit(): s = '_' + s
        return s or '_joint'
    node_to_path = {}
    paths = []
    for j in sorted_joints:
        p = joint_parent[j]
        name = san(nodes[j].get("name", f"joint_{j}"))
        if p is None:
            path = name
        else:
            path = node_to_path[p] + "/" + name
        node_to_path[j] = path
        paths.append(path)
    return paths, node_to_path

# ─── Animation extraction ─────────────────────────────────────────────────────

def extract_animation(gltf, bin_data, anim_idx=0):
    """
    Returns dict: node_idx -> {'T': (times, values), 'R': ..., 'S': ...}
    times: 1D float array, values: (N,3) or (N,4) float array.
    """
    anim = gltf["animations"][anim_idx]
    samplers = anim["samplers"]
    channels = anim["channels"]
    result = {}
    for ch in channels:
        si = ch["sampler"]
        target = ch["target"]
        node_idx = target.get("node")
        path = target["path"]
        if node_idx is None: continue
        samp = samplers[si]
        times = acc_data(gltf, bin_data, samp["input"]).flatten()
        vals  = acc_data(gltf, bin_data, samp["output"])
        interp = samp.get("interpolation", "LINEAR")
        if node_idx not in result:
            result[node_idx] = {}
        key = path[0].upper()  # T, R, or S
        result[node_idx][key] = (times, vals, interp)
    return result

def sample_channel(times, vals, interp, t):
    """Sample a channel at time t (LINEAR or STEP, no cubicspline)."""
    if len(times) == 1:
        return vals[0]
    if t <= times[0]: return vals[0]
    if t >= times[-1]: return vals[-1]
    idx = np.searchsorted(times, t, side='right') - 1
    idx = int(np.clip(idx, 0, len(times)-2))
    if interp == "STEP":
        return vals[idx]
    t0, t1 = times[idx], times[idx+1]
    alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
    v0, v1 = vals[idx], vals[idx+1]
    if v0.shape[0] == 4:  # quaternion — slerp
        return slerp(v0, v1, alpha)
    return v0 + alpha * (v1 - v0)  # lerp

def slerp(q0, q1, t):
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = np.dot(q0, q1)
    if dot < 0: q1 = -q1; dot = -dot
    dot = min(dot, 1.0)
    if dot > 0.9995:
        return (q0 + t*(q1-q0)) / np.linalg.norm(q0 + t*(q1-q0))
    theta = math.acos(dot)
    sin_t = math.sin(theta)
    s0 = math.sin((1-t)*theta)/sin_t
    s1 = math.sin(t*theta)/sin_t
    return s0*q0 + s1*q1

def get_node_trs(gltf_node):
    """Default TRS from a glTF node dict."""
    t = np.array(gltf_node.get("translation",[0,0,0]), dtype=np.float64)
    r = np.array(gltf_node.get("rotation",[0,0,0,1]), dtype=np.float64)
    s = np.array(gltf_node.get("scale",[1,1,1]), dtype=np.float64)
    return t, r, s

def build_anim_frames(anim_data, sorted_joints, joint_parent, parent_map, gltf):
    """
    Build per-frame USD joint-local TRS for N_FRAMES frames.

    Root USD joints store their full glTF world matrix (accounts for non-joint
    ancestor nodes such as the Blender armature node).
    Non-root USD joints store inv(parent_world) @ child_world.

    This ensures the stored TRS matches the coordinate space expected by USD
    UsdSkel, which multiplies up the joint hierarchy to recover world transforms.
    """
    nodes = gltf["nodes"]
    N = len(sorted_joints)

    # All glTF nodes needed: joints + every ancestor up to the scene root
    needed = set()
    for j in sorted_joints:
        n = j
        while n is not None:
            needed.add(n)
            n = parent_map[n]
    needed_order = _topo_sort_nodes(needed, parent_map)

    T_all = np.zeros((N_FRAMES, N, 3))
    R_all = np.zeros((N_FRAMES, N, 4))
    S_all = np.ones((N_FRAMES, N, 3))

    for fi in range(N_FRAMES):
        t_sec = fi / FPS

        # Build per-frame glTF world matrices for all needed nodes
        frame_world = {}
        for nidx in needed_order:
            nd = nodes[nidx]
            rest_t, rest_r, rest_s = get_node_trs(nd)
            if nidx in anim_data:
                ch = anim_data[nidx]
                T = sample_channel(*ch['T'], t_sec) if 'T' in ch else rest_t
                R = sample_channel(*ch['R'], t_sec) if 'R' in ch else rest_r
                S = sample_channel(*ch['S'], t_sec) if 'S' in ch else rest_s
            else:
                T, R, S = rest_t, rest_r, rest_s
            R = R / (np.linalg.norm(R) or 1.0)
            local_m = trs_to_mat(T, R, S)
            p = parent_map[nidx]
            frame_world[nidx] = (frame_world[p] @ local_m) \
                if (p is not None and p in frame_world) else local_m

        # Convert glTF world matrices → USD joint-local TRS
        for si, nidx in enumerate(sorted_joints):
            world_j = frame_world[nidx]
            p_nidx = joint_parent[nidx]
            if p_nidx is None:
                # Root joint: USD local == glTF world (no USD parent)
                local_usd = world_j
            else:
                local_usd = np.linalg.inv(frame_world[p_nidx]) @ world_j
            t_v, r_v, s_v = mat_to_trs(local_usd)
            T_all[fi, si] = t_v
            R_all[fi, si] = r_v
            S_all[fi, si] = s_v

    return T_all, R_all, S_all

# ─── Texture extraction ──────────────────────────────────────────────────────

def extract_textures(gltf, bin_data, out_dir):
    """Extract embedded images to files. Returns {image_index: path}."""
    out_dir.mkdir(exist_ok=True)
    paths = {}
    for i, img in enumerate(gltf.get("images",[])):
        if "bufferView" in img:
            bv = gltf["bufferViews"][img["bufferView"]]
            off = bv.get("byteOffset",0)
            raw = bin_data[off:off+bv["byteLength"]]
            mime = img.get("mimeType","image/png")
            ext = ".jpg" if "jpeg" in mime else ".png"
            fname = img.get("name", f"tex_{i}") + ext
            fname = re.sub(r'[^\w.\-]','_', fname)
            fpath = out_dir / fname
            with open(fpath,"wb") as f: f.write(raw)
            paths[i] = str(fpath)
        elif "uri" in img:
            paths[i] = img["uri"]
    return paths

# ─── Mesh extraction ─────────────────────────────────────────────────────────

def extract_mesh(gltf, bin_data, mesh_idx, skin_joints_ordered,
                 all_sorted_joints, node_to_sorted_idx):
    """
    Returns dict with pos, normals, texcoord, indices, joint_indices, weights
    with joint indices remapped to the global sorted skeleton.
    """
    mesh = gltf["meshes"][mesh_idx]
    prim = mesh["primitives"][0]
    attrs = prim["attributes"]
    pos  = acc_data(gltf, bin_data, attrs["POSITION"])[:,:3]
    norm = acc_data(gltf, bin_data, attrs.get("NORMAL"))[:,:3] \
           if "NORMAL" in attrs else np.zeros_like(pos)
    uv   = acc_data(gltf, bin_data, attrs.get("TEXCOORD_0"))[:,:2] \
           if "TEXCOORD_0" in attrs else np.zeros((len(pos),2))
    ji_raw = acc_int(gltf, bin_data, attrs["JOINTS_0"])   # (V,4) local skin indices
    wt     = acc_data(gltf, bin_data, attrs["WEIGHTS_0"]) # (V,4)
    indices = acc_int(gltf, bin_data, prim["indices"]).flatten() if "indices" in prim \
              else np.arange(len(pos))

    # Remap joint indices: local skin index → global sorted skeleton index
    ji_remapped = np.zeros_like(ji_raw)
    for v in range(len(pos)):
        for k in range(4):
            local_idx = int(ji_raw[v,k])
            if local_idx < len(skin_joints_ordered):
                node_idx = skin_joints_ordered[local_idx]
                ji_remapped[v,k] = node_to_sorted_idx.get(node_idx, 0)
    return dict(pos=pos.astype(np.float32), norm=norm.astype(np.float32),
                uv=uv.astype(np.float32), indices=indices.astype(np.int32),
                joints=ji_remapped.astype(np.int32),
                weights=wt.astype(np.float32))

# ─── Check rest vs bind pose ─────────────────────────────────────────────────

def check_rest_bind(sorted_joints, world_mats, ibm_array, body_skin_joints, body_mesh_world):
    """
    Verify bind_world ≈ rest_world for all joints (C ≈ I expected).
    bind_world = body_mesh_world @ inv(IBM)  — correct formula.
    """
    body_idx = {nidx: i for i, nidx in enumerate(body_skin_joints)}
    C_list = []
    for j in sorted_joints[:20]:
        if j not in body_idx: continue
        bi = body_idx[j]
        ibm = ibm_array[bi].reshape(4,4, order='F')
        bind_w = body_mesh_world @ np.linalg.inv(ibm)
        rest_w = world_mats[j]
        try:
            C = bind_w @ np.linalg.inv(rest_w)
            C_list.append(C)
        except np.linalg.LinAlgError:
            continue
    if not C_list: return None
    C0 = C_list[0]
    diffs = [np.linalg.norm(C - C0) for C in C_list[1:]]
    max_diff = max(diffs) if diffs else 0.0
    print(f"  rest/bind check: max C deviation across first 20 joints: {max_diff:.2e}")
    is_identity = np.linalg.norm(C0 - np.eye(4)) < 1e-3
    if is_identity:
        print("  → rest == bind ✓")
        return None
    print(f"  → rest ≠ bind. C_global:\n{C0}")
    if max_diff < 1e-3:
        print("  → C_global is consistent across joints.")
        return C0
    print("  WARNING: C_global NOT consistent across joints")
    return None

# ─── USD helpers ─────────────────────────────────────────────────────────────

def np_to_gfmat4d(m):
    """numpy 4x4 row-major → GfMatrix4d."""
    return Gf.Matrix4d(*[float(v) for v in m.flatten()])

def np_to_gfmat4f(m):
    return Gf.Matrix4f(*[float(v) for v in m.flatten()])

# ─── Material building ────────────────────────────────────────────────────────

def build_material(stage, mat_path, mat_def, tex_paths, gltf, work_dir):
    """Build a UsdPreviewSurface material."""
    material = UsdShade.Material.Define(stage, mat_path)
    pbr  = mat_def.get("pbrMetallicRoughness",{})

    shader_path = mat_path + "/PBRShader"
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    roughness_val = float(pbr.get("roughnessFactor", 0.7))
    shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)

    # Base color texture
    if "baseColorTexture" in pbr:
        tex_idx = pbr["baseColorTexture"]["index"]
        img_idx = gltf["textures"][tex_idx]["source"]
        tpath = tex_paths.get(img_idx)
        if tpath:
            st_reader = _add_uv_reader(stage, mat_path, "stReader")
            bc_tex = _add_texture(stage, mat_path, "baseColorTex",
                                  tpath, st_reader, "rgb", fallback=(0.8,0.8,0.8,1))
            shader.CreateInput("diffuseColor",
                               Sdf.ValueTypeNames.Color3f).ConnectToSource(
                                   bc_tex.ConnectableAPI(), "rgb")
    else:
        bc = pbr.get("baseColorFactor",[0.8,0.8,0.8,1])
        shader.CreateInput("diffuseColor",
                           Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*bc[:3]))

    # Roughness from metallicRoughness G channel
    if "metallicRoughnessTexture" in pbr:
        tex_idx = pbr["metallicRoughnessTexture"]["index"]
        img_idx = gltf["textures"][tex_idx]["source"]
        tpath = tex_paths.get(img_idx)
        if tpath:
            st_reader = _get_or_add_uv_reader(stage, mat_path)
            mr_tex = _add_texture(stage, mat_path, "mrTex",
                                  tpath, st_reader, "g", fallback=(0,roughness_val,0,1))
            shader.CreateInput("roughness",
                               Sdf.ValueTypeNames.Float).ConnectToSource(
                                   mr_tex.ConnectableAPI(), "r")
    else:
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness_val)

    # Normal map
    if "normalTexture" in mat_def:
        tex_idx = mat_def["normalTexture"]["index"]
        img_idx = gltf["textures"][tex_idx]["source"]
        tpath = tex_paths.get(img_idx)
        if tpath:
            st_reader = _get_or_add_uv_reader(stage, mat_path)
            nm_tex = _add_texture(stage, mat_path, "normalTex",
                                  tpath, st_reader, "rgb", fallback=(0,0,1,1),
                                  color_space="raw")
            shader.CreateInput("normal",
                               Sdf.ValueTypeNames.Normal3f).ConnectToSource(
                                   nm_tex.ConnectableAPI(), "rgb")

    # Wire output
    pbr_out = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(pbr_out)
    return material

def _add_uv_reader(stage, mat_path, name):
    r = UsdShade.Shader.Define(stage, mat_path + f"/{name}")
    r.CreateIdAttr("UsdPrimvarReader_float2")
    r.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    r.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    return r

def _get_or_add_uv_reader(stage, mat_path):
    p = stage.GetPrimAtPath(mat_path + "/stReader")
    if p.IsValid():
        return UsdShade.Shader(p)
    return _add_uv_reader(stage, mat_path, "stReader")

def _add_texture(stage, mat_path, name, file_path, st_reader,
                 channel, fallback, color_space="sRGB"):
    tex = UsdShade.Shader.Define(stage, mat_path + f"/{name}")
    tex.CreateIdAttr("UsdUVTexture")
    # Make path relative to WORK_DIR
    rel = os.path.relpath(file_path, str(WORK_DIR))
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(rel)
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        st_reader.ConnectableAPI(), "result")
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    if color_space == "raw":
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
    else:
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    fb = Gf.Vec4f(*fallback) if len(fallback)==4 else Gf.Vec4f(*fallback,1)
    tex.CreateInput("fallback", Sdf.ValueTypeNames.Float4).Set(fb)
    # Named outputs
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    tex.CreateOutput("r",   Sdf.ValueTypeNames.Float)
    tex.CreateOutput("g",   Sdf.ValueTypeNames.Float)
    tex.CreateOutput("a",   Sdf.ValueTypeNames.Float)
    return tex

# ─── Mesh authoring ──────────────────────────────────────────────────────────

def author_mesh(stage, prim_path, mesh_data, material, skel_path):
    """Write a skinned UsdGeom.Mesh and bind it."""
    usd_mesh = UsdGeom.Mesh.Define(stage, prim_path)
    pos = mesh_data["pos"]
    idx = mesh_data["indices"]
    n_tris = len(idx) // 3

    pts = pos.astype(np.float64)
    usd_mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(float(p[0]),float(p[1]),float(p[2])) for p in pts]))
    usd_mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3]*n_tris))
    usd_mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(idx.tolist()))
    usd_mesh.CreateSubdivisionSchemeAttr("none")

    # Normals
    norm = mesh_data["norm"]
    if len(norm):
        nrm = norm.astype(np.float64)
        usd_mesh.CreateNormalsAttr(
            Vt.Vec3fArray([Gf.Vec3f(float(n[0]),float(n[1]),float(n[2])) for n in nrm]))
        usd_mesh.SetNormalsInterpolation("vertex")

    # UVs
    uv = mesh_data["uv"]
    pv_api = UsdGeom.PrimvarsAPI(usd_mesh.GetPrim())
    if len(uv):
        # Flip V for USD/glTF convention
        uv_flipped = uv.copy(); uv_flipped[:,1] = 1.0 - uv[:,1]
        st_pv = pv_api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex")
        st_pv.Set(Vt.Vec2fArray([Gf.Vec2f(float(u),float(v)) for u,v in uv_flipped.astype(np.float64)]))

    # Skinning primvars
    jt = mesh_data["joints"].flatten().tolist()
    wt = mesh_data["weights"].flatten().tolist()
    ji_pv = pv_api.CreatePrimvar("skel:jointIndices", Sdf.ValueTypeNames.IntArray, "vertex")
    ji_pv.Set(Vt.IntArray(jt)); ji_pv.SetElementSize(4)
    jw_pv = pv_api.CreatePrimvar("skel:jointWeights", Sdf.ValueTypeNames.FloatArray, "vertex")
    jw_pv.Set(Vt.FloatArray([float(w) for w in wt])); jw_pv.SetElementSize(4)

    # Material binding
    UsdShade.MaterialBindingAPI.Apply(usd_mesh.GetPrim()).Bind(material)

    # Skel binding
    binding = UsdSkel.BindingAPI.Apply(usd_mesh.GetPrim())
    binding.CreateSkeletonRel().SetTargets([Sdf.Path(skel_path)])
    binding.CreateGeomBindTransformAttr(Gf.Matrix4d(1))  # identity

    return usd_mesh

# ─── Main build ──────────────────────────────────────────────────────────────

def main():
    print(f"Reading {GLB_IN}…")
    gltf, bin_data = read_glb(GLB_IN)

    # ── Identify skins ──────────────────────────────────────────────────────
    skins = gltf["skins"]
    body_skin_idx = max(range(len(skins)), key=lambda i: len(skins[i]["joints"]))
    eyes_skin_idx = min(range(len(skins)), key=lambda i: len(skins[i]["joints"]))
    body_skin = skins[body_skin_idx]
    eyes_skin = skins[eyes_skin_idx]
    body_skin_joints = body_skin["joints"]   # node indices, original order
    eyes_skin_joints = eyes_skin["joints"]

    print(f"Body skin: {len(body_skin_joints)} joints, Eyes skin: {len(eyes_skin_joints)} joints")

    # ── Node hierarchy ──────────────────────────────────────────────────────
    parent = build_parents(gltf)
    world_mats = compute_world_mats(gltf, parent)

    # ── Topo-sort body joints ───────────────────────────────────────────────
    print("Topological sort of joints…")
    sorted_joints, joint_parent = topo_sort_joints(body_skin_joints, parent)
    joint_paths, node_to_path = joint_usd_paths(sorted_joints, joint_parent, gltf)
    node_to_sorted_idx = {nidx: si for si, nidx in enumerate(sorted_joints)}
    N_JOINTS = len(sorted_joints)
    print(f"  {N_JOINTS} joints sorted. Root(s): {[p for p in joint_paths if '/' not in p][:5]}")

    # ── Extract IBMs ─────────────────────────────────────────────────────────
    ibm_raw = acc_data(gltf, bin_data, body_skin["inverseBindMatrices"])  # (N,16)
    # body_skin_joints[i] corresponds to ibm_raw[i]
    body_joint_to_ibm_idx = {nidx: i for i, nidx in enumerate(body_skin_joints)}

    # Build per-sorted-joint IBM (4x4, column-major from glTF)
    ibm_sorted = []
    for nidx in sorted_joints:
        ii = body_joint_to_ibm_idx[nidx]
        ibm = ibm_raw[ii].reshape(4,4, order='F')
        ibm_sorted.append(ibm)

    # bind_world = inv(IBM)
    bind_worlds = [np.linalg.inv(ibm) for ibm in ibm_sorted]

    # ── Rest vs bind check ───────────────────────────────────────────────────
    print("Checking rest vs bind pose…")
    C_global = check_rest_bind(sorted_joints, world_mats, ibm_raw, body_skin_joints)

    # ── Compute rest local transforms for USD (from bind pose) ───────────────
    # restTransforms: LOCAL matrix of each joint relative to its parent
    # Derive from bind_world (inv IBM) to ensure rest == bind in USD
    rest_locals_mat = []
    for si, nidx in enumerate(sorted_joints):
        p_nidx = joint_parent[nidx]
        if p_nidx is None:
            local_m = bind_worlds[si]
        else:
            p_si = node_to_sorted_idx[p_nidx]
            local_m = np.linalg.inv(bind_worlds[p_si]) @ bind_worlds[si]
        rest_locals_mat.append(local_m)

    # ── Extract animation ────────────────────────────────────────────────────
    print("Extracting animation…")
    anim_data = extract_animation(gltf, bin_data, 0)
    print(f"  {len(anim_data)} nodes have animation channels")

    print("Sampling 101 frames…")
    T_all, R_all, S_all = build_anim_frames(anim_data, sorted_joints, gltf)
    print(f"  T_all: {T_all.shape}, R_all: {R_all.shape}")

    # ── Extract textures ─────────────────────────────────────────────────────
    print("Extracting textures…")
    tex_paths = extract_textures(gltf, bin_data, WORK_DIR / "textures")
    print(f"  {len(tex_paths)} textures → {WORK_DIR/'textures'}")

    # ── Extract meshes ───────────────────────────────────────────────────────
    # Find which mesh index belongs to body vs eyes
    nodes = gltf["nodes"]
    body_mesh_node = next(nd for nd in nodes
                          if nd.get("skin") == body_skin_idx and "mesh" in nd)
    eyes_mesh_node = next(nd for nd in nodes
                          if nd.get("skin") == eyes_skin_idx and "mesh" in nd)

    # Check mesh node world transforms
    body_nd_idx = nodes.index(body_mesh_node)
    eyes_nd_idx = nodes.index(eyes_mesh_node)
    body_mesh_world = world_mats[body_nd_idx]
    eyes_mesh_world = world_mats[eyes_nd_idx]
    print(f"Body mesh node world transform (should be ~identity):\n{body_mesh_world}")

    print("Extracting meshes…")
    body_data = extract_mesh(gltf, bin_data, body_mesh_node["mesh"],
                             body_skin_joints, sorted_joints, node_to_sorted_idx)
    eyes_data = extract_mesh(gltf, bin_data, eyes_mesh_node["mesh"],
                             eyes_skin_joints, sorted_joints, node_to_sorted_idx)
    print(f"  Body: {len(body_data['pos'])} verts, {len(body_data['indices'])//3} tris")
    print(f"  Eyes: {len(eyes_data['pos'])} verts, {len(eyes_data['indices'])//3} tris")

    # ── Compute Y bounds across all frames for floor anchoring ───────────────
    print("Computing animated bounds (all 101 frames)…")
    min_y_global = float('inf')
    max_extent = 0.0
    # Sample frames 0,25,50,75,100 for speed
    for fi in [0, 12, 25, 37, 50, 62, 75, 87, 100]:
        frame_mins, frame_maxs = _skinned_bounds(
            fi, body_data, sorted_joints, bind_worlds, T_all, R_all, S_all,
            joint_parent, node_to_sorted_idx)
        min_y_global = min(min_y_global, frame_mins[1])
        ext = max(frame_maxs) - min(frame_mins)
        max_extent = max(max_extent, ext)
    print(f"  Min Y across animation: {min_y_global:.4f}")
    print(f"  Approx max extent: {max_extent:.4f}")

    # Scale to ~0.5m real-world span; Y-offset to keep above ground
    raw_scale = 0.5 / max(max_extent, 0.01)
    y_offset  = -min_y_global * raw_scale + 0.01  # 1cm above floor
    print(f"  Scale: {raw_scale:.4f}, Y offset: {y_offset:.4f}")

    # ── Build USD stage ──────────────────────────────────────────────────────
    print(f"\nBuilding USD stage → {USDA_OUT}")
    if os.path.exists(USDA_OUT): os.remove(USDA_OUT)
    stage = Usd.Stage.CreateNew(USDA_OUT)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    stage.SetTimeCodesPerSecond(FPS)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(N_FRAMES - 1)
    stage.SetMetadata("metersPerUnit", 1.0)

    # Root
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    # Invisible ground anchor for AR floor tracking
    ground = UsdGeom.Mesh.Define(stage, "/World/GroundAnchor")
    ground.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(-.01,0,-.01),
                                            Gf.Vec3f( .01,0,-.01),
                                            Gf.Vec3f( .01,0, .01),
                                            Gf.Vec3f(-.01,0, .01)]))
    ground.CreateFaceVertexCountsAttr(Vt.IntArray([4]))
    ground.CreateFaceVertexIndicesAttr(Vt.IntArray([0,1,2,3]))
    ground.CreateSubdivisionSchemeAttr("none")
    UsdGeom.Imageable(ground).MakeInvisible()

    # Octopus group (scale + Y lift)
    oct_xf = UsdGeom.Xform.Define(stage, "/World/Octopus")
    oct_xf.AddTranslateOp().Set(Gf.Vec3d(0, y_offset, 0))
    oct_xf.AddScaleOp().Set(Gf.Vec3d(raw_scale, raw_scale, raw_scale))

    # Materials
    mats_scope = UsdGeom.Scope.Define(stage, "/World/Materials")
    gltf_mats = gltf.get("materials",[])
    body_mat_def = next((m for m in gltf_mats if m.get("name")=="octopus_body"), gltf_mats[0])
    eyes_mat_def = next((m for m in gltf_mats if m.get("name")=="octopus_eyes"), gltf_mats[-1])
    body_mat = build_material(stage, "/World/Materials/BodyMat",
                              body_mat_def, tex_paths, gltf, WORK_DIR)
    eyes_mat = build_material(stage, "/World/Materials/EyesMat",
                              eyes_mat_def, tex_paths, gltf, WORK_DIR)

    # SkelRoot
    skel_root = UsdSkel.Root.Define(stage, "/World/Octopus/SkelRoot")
    skel_root_binding = UsdSkel.BindingAPI.Apply(skel_root.GetPrim())

    # Skeleton
    skel_prim_path = "/World/Octopus/SkelRoot/Skeleton"
    skel = UsdSkel.Skeleton.Define(stage, skel_prim_path)
    skel.CreateJointsAttr(Vt.TokenArray(joint_paths))

    # restTransforms (local, derived from bind-world so rest == bind)
    rest_xforms = []
    for mat in rest_locals_mat:
        rest_xforms.append(np_to_gfmat4d(mat))
    skel.CreateRestTransformsAttr(Vt.Matrix4dArray(rest_xforms))

    # bindTransforms (world, from IBM inverse)
    bind_xforms = [np_to_gfmat4d(m) for m in bind_worlds]
    skel.CreateBindTransformsAttr(Vt.Matrix4dArray(bind_xforms))

    # SkelAnimation
    anim_path = "/World/Octopus/SkelRoot/SkelAnim"
    skel_anim = UsdSkel.Animation.Define(stage, anim_path)
    skel_anim.CreateJointsAttr(Vt.TokenArray(joint_paths))

    print("Writing animation time samples…")
    trans_attr  = skel_anim.CreateTranslationsAttr()
    rot_attr    = skel_anim.CreateRotationsAttr()
    scale_attr  = skel_anim.CreateScalesAttr()

    for fi in range(N_FRAMES):
        tc = float(fi)
        T = T_all[fi]  # (N_JOINTS, 3)
        R = R_all[fi]  # (N_JOINTS, 4) xyzw
        S = S_all[fi]  # (N_JOINTS, 3)

        trans_arr  = Vt.Vec3fArray([Gf.Vec3f(*t.tolist()) for t in T])
        rot_arr    = Vt.QuatfArray([Gf.Quatf(float(r[3]),
                                              float(r[0]),
                                              float(r[1]),
                                              float(r[2])) for r in R])
        scale_arr  = Vt.Vec3hArray([Gf.Vec3h(float(s[0]),
                                              float(s[1]),
                                              float(s[2])) for s in S])
        trans_attr.Set(trans_arr, tc)
        rot_attr.Set(rot_arr, tc)
        scale_attr.Set(scale_arr, tc)

    # Wire animation source
    skel_binding = UsdSkel.BindingAPI.Apply(skel.GetPrim())
    skel_binding.CreateAnimationSourceRel().SetTargets([Sdf.Path(anim_path)])

    # Meshes
    body_mesh_prim = author_mesh(stage, "/World/Octopus/SkelRoot/BodyMesh",
                                  body_data, body_mat, skel_prim_path)
    eyes_mesh_prim = author_mesh(stage, "/World/Octopus/SkelRoot/EyesMesh",
                                  eyes_data, eyes_mat, skel_prim_path)

    stage.Save()
    print(f"Saved {USDA_OUT}  ({os.path.getsize(USDA_OUT)//1024} KB)")

    # ── Verify animation sample count in USDA ───────────────────────────────
    ns = trans_attr.GetNumTimeSamples()
    print(f"\nAnimation: translations have {ns} time samples (expected {N_FRAMES})")
    assert ns == N_FRAMES, f"Expected {N_FRAMES} samples, got {ns}"

    # ── Convert to binary USDC ───────────────────────────────────────────────
    print(f"\nConverting to binary USDC → {USDC_OUT}…")
    if os.path.exists(USDC_OUT): os.remove(USDC_OUT)
    stage2 = Usd.Stage.Open(USDA_OUT)
    stage2.Export(USDC_OUT)
    del stage2

    # Verify sample count survives binary conversion
    stage3 = Usd.Stage.Open(USDC_OUT)
    anim_back = UsdSkel.Animation(stage3.GetPrimAtPath(anim_path))
    ns2 = anim_back.GetTranslationsAttr().GetNumTimeSamples()
    print(f"USDC animation: translations have {ns2} time samples (expected {N_FRAMES})")
    assert ns2 == N_FRAMES, f"USDC binary lost samples: {ns2}"
    del stage3

    print(f"USDC size: {os.path.getsize(USDC_OUT)//1024} KB")

    # ── Package as USDZ ─────────────────────────────────────────────────────
    print(f"\nPackaging → {USDZ_OUT}…")
    if os.path.exists(USDZ_OUT): os.remove(USDZ_OUT)

    # Gather assets: usdc + textures
    assets = [USDC_OUT]
    for tp in tex_paths.values():
        if os.path.exists(tp): assets.append(tp)

    # UsdUtils.CreateNewUsdzPackage wants absolute paths
    abs_assets = [os.path.abspath(a) for a in assets]
    abs_usdz   = os.path.abspath(USDZ_OUT)
    result = UsdUtils.CreateNewUsdzPackage(
        Sdf.AssetPath(abs_assets[0]),
        abs_usdz)
    if not result:
        # Fallback: manual ZIP with correct alignment
        print("  UsdUtils packaging failed, building USDZ manually…")
        _make_usdz_manual(abs_assets, abs_usdz)

    sz = os.path.getsize(USDZ_OUT)
    print(f"USDZ written: {sz//1024} KB")

    # ── Verify USDZ ─────────────────────────────────────────────────────────
    print("\n=== USDZ Verification ===")
    verify_usdz(USDZ_OUT, anim_path, skel_prim_path,
                sorted_joints, bind_worlds, body_data)

    print("\n✓ Pipeline complete.")
    return USDZ_OUT

# ─── Manual USDZ builder (64-byte aligned, uncompressed) ─────────────────────

def _make_usdz_manual(asset_paths, out_path):
    """Build USDZ zip with 64-byte-aligned uncompressed entries."""
    import zipfile as zf
    ALIGN = 64
    with open(out_path, "wb") as fout:
        zw = zf.ZipFile(fout, "w", compression=zf.ZIP_STORED, allowZip64=False)
        for i, src in enumerate(asset_paths):
            arcname = os.path.basename(src)
            data = open(src,"rb").read()
            # We need to pad to reach 64-byte alignment for the data start.
            # Local file header = 30 + len(arcname) bytes (no extra field yet).
            # We'll add extra field padding to hit alignment.
            hdr_base = 30 + len(arcname.encode())
            # Current position in the file
            current_pos = fout.tell() + hdr_base
            needed = (ALIGN - (current_pos % ALIGN)) % ALIGN
            zinfo = zf.ZipInfo(arcname)
            zinfo.compress_type = zf.ZIP_STORED
            zinfo.extra = b'\x00' * needed
            zw.writestr(zinfo, data)
        zw.close()


def _skinned_bounds(fi, mesh_data, sorted_joints, bind_worlds,
                    T_all, R_all, S_all, joint_parent, node_to_sorted_idx):
    """Approximate skinned bounds at frame fi (sample every 50th vertex)."""
    N = len(sorted_joints)
    # Build animated world mats
    anim_worlds = [None]*N
    for si in range(N):
        T = T_all[fi, si]
        R = R_all[fi, si]  # xyzw
        S = S_all[fi, si]
        lm = trs_to_mat(T, R, S)
        p_nidx_si = None
        for nidx, idx in node_to_sorted_idx.items():
            if idx == si:
                p_nidx = joint_parent.get(nidx) if isinstance(joint_parent, dict) else None
                if p_nidx is not None and p_nidx in node_to_sorted_idx:
                    p_nidx_si = node_to_sorted_idx[p_nidx]
                break
        if p_nidx_si is not None and anim_worlds[p_nidx_si] is not None:
            anim_worlds[si] = anim_worlds[p_nidx_si] @ lm
        else:
            anim_worlds[si] = lm

    # Skin sample vertices
    pos = mesh_data["pos"]
    ji  = mesh_data["joints"]
    wt  = mesh_data["weights"]
    step = max(1, len(pos)//200)
    pts = []
    for vi in range(0, len(pos), step):
        p = np.array([*pos[vi], 1.0])
        sp = np.zeros(3)
        for k in range(4):
            w = float(wt[vi,k])
            if w < 1e-5: continue
            jsi = int(ji[vi,k])
            skin_mat = anim_worlds[jsi] @ np.linalg.inv(bind_worlds[jsi])
            sp += w * (skin_mat @ p)[:3]
        pts.append(sp)
    pts = np.array(pts)
    return pts.min(axis=0), pts.max(axis=0)


# ─── USDZ verification ───────────────────────────────────────────────────────

def verify_usdz(usdz_path, anim_path, skel_path, sorted_joints,
                bind_worlds, body_data):
    # 1. ZIP integrity
    with zipfile.ZipFile(usdz_path) as z:
        bad = z.testzip()
        print(f"ZIP integrity: {'OK' if bad is None else 'FAIL: '+bad}")
        for info in z.infolist():
            print(f"  {info.filename}  compress={info.compress_type}  "
                  f"size={info.file_size//1024}KB")

    # 2. 64-byte alignment
    print("64-byte alignment check:")
    with open(usdz_path,"rb") as f:
        raw = f.read()
    ok = _check_zip_alignment(raw)
    print(f"  Alignment: {'OK' if ok else 'WARN — not all entries 64-byte aligned'}")

    # 3. Reopen and check animation
    stage = Usd.Stage.Open(usdz_path)
    anim_back = UsdSkel.Animation(stage.GetPrimAtPath(anim_path))
    if not anim_back:
        print("ERROR: SkelAnimation prim not found in packaged USDZ!")
        return
    ns = anim_back.GetTranslationsAttr().GetNumTimeSamples()
    print(f"Animation time samples in packaged USDZ: {ns} (expected {N_FRAMES})")
    if ns < N_FRAMES:
        print("WARNING: fewer samples than expected — animation may not play!")

    # 4. Skinning deformation check (bounding box changes across frames)
    print("Skinning deformation check across frames [0,25,50,75,100]…")
    cache = UsdSkel.Cache()
    root_prim = stage.GetPrimAtPath("/World/Octopus/SkelRoot")
    cache.Populate(UsdSkel.Root(root_prim), Usd.TraverseInstanceProxies())
    body_prim = stage.GetPrimAtPath("/World/Octopus/SkelRoot/BodyMesh")
    skel_prim = stage.GetPrimAtPath(skel_path)
    query = cache.GetSkelQuery(UsdSkel.Skeleton(skel_prim)) if skel_prim else None

    bboxes = []
    for fi in [0, 25, 50, 75, 100]:
        tc = Usd.TimeCode(fi)
        # Try ComputeSkinnedPoints if possible
        try:
            binding_query = cache.GetSkinningQuery(body_prim)
            if binding_query:
                pts_attr = UsdGeom.PointBased(body_prim).GetPointsAttr()
                pts = pts_attr.Get(tc)
                if pts:
                    arr = np.array([[p[0],p[1],p[2]] for p in pts])
                    mn = arr.min(axis=0); mx = arr.max(axis=0)
                    bboxes.append((mn,mx))
                    print(f"  frame {fi:3d}: Y=[{mn[1]:.4f},{mx[1]:.4f}]  "
                          f"X=[{mn[0]:.4f},{mx[0]:.4f}]")
        except Exception as e:
            print(f"  frame {fi}: skinning query failed ({e})")
            break

    if len(bboxes) >= 2:
        y_ranges = [(b[0][1], b[1][1]) for b in bboxes]
        y_span   = max(b[1][1]-b[0][1] for b in bboxes)
        min_y_all = min(b[0][1] for b in bboxes)
        print(f"  Max Y span across frames: {y_span:.4f}")
        print(f"  Min Y (all frames): {min_y_all:.4f}  "
              f"{'OK above ground' if min_y_all > -0.001 else 'WARN: below ground'}")
        if bboxes[0][0][1] != bboxes[2][0][1]:
            print("  ✓ Bounding box changes across frames → animation is live")
        else:
            print("  ⚠ Bounding boxes identical → animation may be static")

    # 5. Material check
    body_prim2 = stage.GetPrimAtPath("/World/Octopus/SkelRoot/BodyMesh")
    binding = UsdShade.MaterialBindingAPI(body_prim2)
    mat = binding.ComputeBoundMaterial()[0]
    if mat:
        surf_out = mat.GetSurfaceOutput()
        print(f"Material bound: {mat.GetPath()}  surface={'wired' if surf_out else 'missing'}")

    del stage

def _check_zip_alignment(data):
    """Check all local file entries are 64-byte aligned."""
    ALIGN = 64
    ok = True
    pos = 0
    while pos < len(data)-4:
        sig = struct.unpack_from("<I", data, pos)[0]
        if sig == 0x04034b50:  # local file header
            fname_len, extra_len = struct.unpack_from("<HH", data, pos+26)
            data_start = pos + 30 + fname_len + extra_len
            if data_start % ALIGN != 0:
                ok = False
                fname = data[pos+30:pos+30+fname_len].decode(errors='replace')
                print(f"    MISALIGNED: {fname} at offset {data_start} (mod64={data_start%64})")
            csize = struct.unpack_from("<I", data, pos+18)[0]
            pos = data_start + csize
        elif sig == 0x02014b50:  # central dir
            break
        else:
            pos += 1
    return ok


# Global ref needed by _skinned_bounds
ibm_sorted_global = []

if __name__ == "__main__":
    # We need to make ibm_sorted_global available to _skinned_bounds
    # It'll be set during main() after IBMs are computed
    import types
    _orig_main = main

    def patched_main():
        global ibm_sorted_global
        print(f"Reading {GLB_IN}…")
        gltf, bin_data = read_glb(GLB_IN)
        skins = gltf["skins"]
        body_skin_idx = max(range(len(skins)), key=lambda i: len(skins[i]["joints"]))
        eyes_skin_idx = min(range(len(skins)), key=lambda i: len(skins[i]["joints"]))
        body_skin = skins[body_skin_idx]
        eyes_skin = skins[eyes_skin_idx]
        body_skin_joints = body_skin["joints"]
        eyes_skin_joints = eyes_skin["joints"]
        parent = build_parents(gltf)
        world_mats = compute_world_mats(gltf, parent)
        sorted_joints, joint_parent = topo_sort_joints(body_skin_joints, parent)
        joint_paths, node_to_path = joint_usd_paths(sorted_joints, joint_parent, gltf)
        node_to_sorted_idx = {nidx: si for si, nidx in enumerate(sorted_joints)}
        N_JOINTS = len(sorted_joints)
        print(f"Body skin: {len(body_skin_joints)} joints, Eyes skin: {len(eyes_skin_joints)} joints")
        print(f"Topological sort: {N_JOINTS} joints. Root(s): {[p for p in joint_paths if '/' not in p][:5]}")

        ibm_raw = acc_data(gltf, bin_data, body_skin["inverseBindMatrices"])
        body_joint_to_ibm_idx = {nidx: i for i, nidx in enumerate(body_skin_joints)}
        ibm_sorted = []
        for nidx in sorted_joints:
            ii = body_joint_to_ibm_idx[nidx]
            ibm = ibm_raw[ii].reshape(4,4, order='F')
            ibm_sorted.append(ibm)
        ibm_sorted_global = ibm_sorted

        # Find body mesh world transform (needed for correct bind_worlds formula).
        # IBM_k = inv(J_bind_world_k) * M_mesh_world  →  J_bind_world_k = M_mesh_world @ inv(IBM_k)
        _nodes_early = gltf["nodes"]
        _body_mesh_node_early = next(nd for nd in _nodes_early
                                     if nd.get("skin") == body_skin_idx and "mesh" in nd)
        _body_nd_idx_early = _nodes_early.index(_body_mesh_node_early)
        body_mesh_world_early = world_mats[_body_nd_idx_early]

        bind_worlds = [body_mesh_world_early @ np.linalg.inv(ibm) for ibm in ibm_sorted]

        print("Checking rest vs bind pose…")
        C_global = check_rest_bind(sorted_joints, world_mats, ibm_raw, body_skin_joints,
                                   body_mesh_world_early)

        rest_locals_mat = []
        for si, nidx in enumerate(sorted_joints):
            p_nidx = joint_parent[nidx]
            if p_nidx is None:
                local_m = bind_worlds[si]
            else:
                p_si = node_to_sorted_idx[p_nidx]
                local_m = np.linalg.inv(bind_worlds[p_si]) @ bind_worlds[si]
            rest_locals_mat.append(local_m)

        print("Extracting animation…")
        anim_data = extract_animation(gltf, bin_data, 0)
        print(f"  {len(anim_data)} nodes have animation channels")
        print("Sampling 101 frames…")
        T_all, R_all, S_all = build_anim_frames(anim_data, sorted_joints,
                                                  joint_parent, parent, gltf)

        print("Extracting textures…")
        tex_paths = extract_textures(gltf, bin_data, WORK_DIR / "textures")
        print(f"  {len(tex_paths)} textures")

        nodes = gltf["nodes"]
        body_mesh_node = next(nd for nd in nodes
                              if nd.get("skin") == body_skin_idx and "mesh" in nd)
        eyes_mesh_node = next(nd for nd in nodes
                              if nd.get("skin") == eyes_skin_idx and "mesh" in nd)
        body_nd_idx = nodes.index(body_mesh_node)
        eyes_nd_idx = nodes.index(eyes_mesh_node)
        body_mesh_world = world_mats[body_nd_idx]
        eyes_mesh_world = world_mats[eyes_nd_idx]
        is_identity = np.allclose(body_mesh_world, np.eye(4), atol=1e-4)
        print(f"Body mesh node world transform {'≈ identity ✓' if is_identity else '≠ identity — baking into vertices'}")

        print("Extracting meshes…")
        body_data = extract_mesh(gltf, bin_data, body_mesh_node["mesh"],
                                 body_skin_joints, sorted_joints, node_to_sorted_idx)
        eyes_data = extract_mesh(gltf, bin_data, eyes_mesh_node["mesh"],
                                 eyes_skin_joints, sorted_joints, node_to_sorted_idx)
        print(f"  Body: {len(body_data['pos'])} verts, {len(body_data['indices'])//3} tris")
        print(f"  Eyes: {len(eyes_data['pos'])} verts, {len(eyes_data['indices'])//3} tris")

        # If mesh world != identity, bake into vertices and adjust IBM
        if not is_identity:
            M = body_mesh_world
            h = np.ones((len(body_data['pos']),1), dtype=np.float32)
            body_data['pos'] = (np.hstack([body_data['pos'],h]) @ M.T.astype(np.float32))[:,:3]
            body_data['norm'] = (body_data['norm'] @ M[:3,:3].T.astype(np.float32))
        M2 = eyes_mesh_world
        if not np.allclose(M2, np.eye(4), atol=1e-4):
            h = np.ones((len(eyes_data['pos']),1), dtype=np.float32)
            eyes_data['pos'] = (np.hstack([eyes_data['pos'],h]) @ M2.T.astype(np.float32))[:,:3]

        print("Computing animated bounds…")
        min_y_global = float('inf'); max_extent = 0.0
        for fi in [0, 12, 25, 37, 50, 62, 75, 87, 100]:
            frame_mins, frame_maxs = _skinned_bounds(
                fi, body_data, sorted_joints, bind_worlds, T_all, R_all, S_all,
                {nidx: joint_parent[nidx] for nidx in sorted_joints},
                node_to_sorted_idx)
            min_y_global = min(min_y_global, float(frame_mins[1]))
            ext = float(max(frame_maxs) - min(frame_mins))
            max_extent = max(max_extent, ext)
        print(f"  Min Y: {min_y_global:.4f}, Max extent: {max_extent:.4f}")
        raw_scale = 0.5 / max(max_extent, 0.01)
        y_offset  = -min_y_global * raw_scale + 0.01

        print(f"  Scale: {raw_scale:.4f}, Y offset: {y_offset:.4f}")

        print(f"\nBuilding USD stage → {USDA_OUT}")
        if os.path.exists(USDA_OUT): os.remove(USDA_OUT)
        stage = Usd.Stage.CreateNew(USDA_OUT)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        stage.SetTimeCodesPerSecond(FPS)
        stage.SetStartTimeCode(0)
        stage.SetEndTimeCode(N_FRAMES - 1)
        stage.SetMetadata("metersPerUnit", 1.0)

        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())

        ground = UsdGeom.Mesh.Define(stage, "/World/GroundAnchor")
        ground.CreatePointsAttr(Vt.Vec3fArray([
            Gf.Vec3f(-.01,0,-.01), Gf.Vec3f(.01,0,-.01),
            Gf.Vec3f(.01,0,.01),   Gf.Vec3f(-.01,0,.01)]))
        ground.CreateFaceVertexCountsAttr(Vt.IntArray([4]))
        ground.CreateFaceVertexIndicesAttr(Vt.IntArray([0,1,2,3]))
        ground.CreateSubdivisionSchemeAttr("none")
        UsdGeom.Imageable(ground).MakeInvisible()

        oct_xf = UsdGeom.Xform.Define(stage, "/World/Octopus")
        oct_xf.AddTranslateOp().Set(Gf.Vec3d(0, float(y_offset), 0))
        oct_xf.AddScaleOp().Set(Gf.Vec3d(float(raw_scale), float(raw_scale), float(raw_scale)))

        UsdGeom.Scope.Define(stage, "/World/Materials")
        gltf_mats = gltf.get("materials",[])
        body_mat_def = next((m for m in gltf_mats if m.get("name")=="octopus_body"), gltf_mats[0])
        eyes_mat_def = next((m for m in gltf_mats if m.get("name")=="octopus_eyes"), gltf_mats[-1])
        body_mat = build_material(stage, "/World/Materials/BodyMat",
                                  body_mat_def, tex_paths, gltf, WORK_DIR)
        eyes_mat = build_material(stage, "/World/Materials/EyesMat",
                                  eyes_mat_def, tex_paths, gltf, WORK_DIR)

        skel_root = UsdSkel.Root.Define(stage, "/World/Octopus/SkelRoot")

        skel_prim_path = "/World/Octopus/SkelRoot/Skeleton"
        skel = UsdSkel.Skeleton.Define(stage, skel_prim_path)
        skel.CreateJointsAttr(Vt.TokenArray(joint_paths))
        skel.CreateRestTransformsAttr(Vt.Matrix4dArray([np_to_gfmat4d(m) for m in rest_locals_mat]))
        skel.CreateBindTransformsAttr(Vt.Matrix4dArray([np_to_gfmat4d(m) for m in bind_worlds]))

        anim_prim_path = "/World/Octopus/SkelRoot/SkelAnim"
        skel_anim = UsdSkel.Animation.Define(stage, anim_prim_path)
        skel_anim.CreateJointsAttr(Vt.TokenArray(joint_paths))

        print("Writing 101 animation time samples…")
        trans_attr = skel_anim.CreateTranslationsAttr()
        rot_attr   = skel_anim.CreateRotationsAttr()
        scale_attr = skel_anim.CreateScalesAttr()

        for fi in range(N_FRAMES):
            tc = float(fi)
            T = T_all[fi]; R = R_all[fi]; S = S_all[fi]
            trans_attr.Set(Vt.Vec3fArray([Gf.Vec3f(*t.tolist()) for t in T]), tc)
            rot_attr.Set(Vt.QuatfArray([Gf.Quatf(float(r[3]),float(r[0]),float(r[1]),float(r[2])) for r in R]), tc)
            scale_attr.Set(Vt.Vec3hArray([Gf.Vec3h(float(s[0]),float(s[1]),float(s[2])) for s in S]), tc)

        skel_binding = UsdSkel.BindingAPI.Apply(skel.GetPrim())
        skel_binding.CreateAnimationSourceRel().SetTargets([Sdf.Path(anim_prim_path)])

        author_mesh(stage, "/World/Octopus/SkelRoot/BodyMesh",
                    body_data, body_mat, skel_prim_path)
        author_mesh(stage, "/World/Octopus/SkelRoot/EyesMesh",
                    eyes_data, eyes_mat, skel_prim_path)

        stage.Save()
        print(f"Saved {USDA_OUT}  ({os.path.getsize(USDA_OUT)//1024} KB)")

        ns = trans_attr.GetNumTimeSamples()
        print(f"Translation samples in USDA: {ns}")
        assert ns == N_FRAMES, f"Expected {N_FRAMES} got {ns}"

        print(f"\nConverting to binary USDC…")
        if os.path.exists(USDC_OUT): os.remove(USDC_OUT)
        s2 = Usd.Stage.Open(USDA_OUT); s2.Export(USDC_OUT); del s2
        s3 = Usd.Stage.Open(USDC_OUT)
        ns2 = UsdSkel.Animation(s3.GetPrimAtPath(anim_prim_path)).GetTranslationsAttr().GetNumTimeSamples()
        print(f"Translation samples in USDC: {ns2}")
        assert ns2 == N_FRAMES, f"USDC lost samples: {ns2}"
        del s3
        print(f"USDC: {os.path.getsize(USDC_OUT)//1024} KB")

        print(f"\nPackaging USDZ…")
        if os.path.exists(USDZ_OUT): os.remove(USDZ_OUT)
        tex_files = list(tex_paths.values())
        # Copy textures into WORK_DIR so relative paths resolve
        for tp in tex_files:
            dst = WORK_DIR / os.path.basename(tp)
            if not dst.exists(): shutil.copy(tp, dst)

        abs_usdc = os.path.abspath(USDC_OUT)
        abs_usdz = os.path.abspath(USDZ_OUT)
        ok = UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(abs_usdc), abs_usdz)
        if not ok or not os.path.exists(USDZ_OUT) or os.path.getsize(USDZ_OUT) < 1000:
            print("  UsdUtils.CreateNewUsdzPackage failed — building manually")
            all_assets = [abs_usdc] + [os.path.abspath(str(WORK_DIR/os.path.basename(tp))) for tp in tex_files if os.path.exists(str(WORK_DIR/os.path.basename(tp)))]
            _make_usdz_manual(all_assets, abs_usdz)

        sz = os.path.getsize(USDZ_OUT)
        print(f"USDZ: {sz//1024} KB  ({sz/1048576:.2f} MB)")

        print("\n=== Verifying packaged USDZ ===")
        verify_usdz(USDZ_OUT, anim_prim_path, skel_prim_path,
                    sorted_joints, bind_worlds, body_data)

        print(f"\n✓ Done → {USDZ_OUT}")

    patched_main()
