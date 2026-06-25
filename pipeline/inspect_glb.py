#!/usr/bin/env python3
"""Inspect GLB: animations, meshes, skins, materials, joint hierarchy."""
import struct, json, sys, math
import numpy as np

GLB_PATH = sys.argv[1] if len(sys.argv) > 1 else "giant_pacific_octopus_swim_A3.glb"

def read_glb(path):
    with open(path, "rb") as f:
        magic, ver, length = struct.unpack("<III", f.read(12))
        assert magic == 0x46546C67 and ver == 2
        chunks = {}
        while True:
            hdr = f.read(8)
            if len(hdr) < 8: break
            clen, ctype = struct.unpack("<II", hdr)
            data = f.read(clen)
            chunks[ctype] = data
    gltf = json.loads(chunks[0x4E4F534A])  # JSON
    bin_data = chunks.get(0x004E4942, b"")  # BIN
    return gltf, bin_data

def get_accessor(gltf, bin_data, idx):
    acc = gltf["accessors"][idx]
    bv = gltf["bufferViews"][acc["bufferView"]]
    dtype_map = {5120:"b",5121:"B",5122:"h",5123:"H",5125:"I",5126:"f"}
    type_size = {"SCALAR":1,"VEC2":2,"VEC3":3,"VEC4":4,"MAT2":4,"MAT3":9,"MAT4":16}
    fmt = dtype_map[acc["componentType"]]
    nc = type_size[acc["type"]]
    item = struct.calcsize(fmt) * nc
    stride = bv.get("byteStride", item)
    offset = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    count = acc["count"]
    vals = []
    for i in range(count):
        row = struct.unpack_from(f"<{nc}{fmt}", bin_data, offset + i*stride)
        if acc.get("normalized"):
            if acc["componentType"] == 5122: row = tuple(max(v/32767,-1) for v in row)
            elif acc["componentType"] == 5121: row = tuple(v/255 for v in row)
            elif acc["componentType"] == 5123: row = tuple(v/65535 for v in row)
        vals.append(row)
    return np.array(vals, dtype=np.float32)

gltf, bin_data = read_glb(GLB_PATH)
print(f"=== GLB: {GLB_PATH} ===")

# Animations
anims = gltf.get("animations", [])
print(f"\n--- Animations ({len(anims)}) ---")
for i, a in enumerate(anims):
    name = a.get("name","<unnamed>")
    chs = a.get("channels",[])
    samps = a.get("samplers",[])
    # time range
    times_all = []
    for s in samps:
        inp = get_accessor(gltf, bin_data, s["input"])
        times_all.extend(inp[:,0].tolist())
    tmin = min(times_all); tmax = max(times_all)
    frame_count = len(set(round(t,6) for t in times_all))
    print(f"  [{i}] '{name}'  channels={len(chs)}  time=[{tmin:.4f},{tmax:.4f}]s  unique_times≈{frame_count}")

# Meshes
meshes = gltf.get("meshes", [])
print(f"\n--- Meshes ({len(meshes)}) ---")
for i, m in enumerate(meshes):
    for j, prim in enumerate(m.get("primitives",[])):
        pos_idx = prim["attributes"].get("POSITION")
        vc = gltf["accessors"][pos_idx]["count"] if pos_idx is not None else 0
        mat_idx = prim.get("material")
        mat_name = gltf["materials"][mat_idx].get("name","?") if mat_idx is not None else "none"
        print(f"  mesh[{i}] prim[{j}] '{m.get('name','?')}' verts={vc} mat='{mat_name}'")
        attrs = list(prim["attributes"].keys())
        print(f"    attributes: {attrs}")

# Skins
skins = gltf.get("skins", [])
print(f"\n--- Skins ({len(skins)}) ---")
for i, sk in enumerate(skins):
    joints = sk.get("joints",[])
    jnames = [gltf["nodes"][j].get("name","?") for j in joints]
    print(f"  skin[{i}] '{sk.get('name','?')}' joints={len(joints)}")
    print(f"    first 10 joints: {jnames[:10]}")
    print(f"    last 5 joints:   {jnames[-5:]}")
    # Find eye joints
    eye_joints = [(jnames[k], joints[k]) for k in range(len(joints)) if 'eye' in jnames[k].lower() or 'Eye' in jnames[k]]
    if eye_joints:
        print(f"    eye joints: {eye_joints}")

# Nodes referencing skins / mesh
nodes = gltf.get("nodes",[])
print(f"\n--- Skinned mesh nodes ---")
for i, nd in enumerate(nodes):
    if "skin" in nd and "mesh" in nd:
        skin_idx = nd["skin"]
        mesh_idx = nd["mesh"]
        mesh_name = gltf["meshes"][mesh_idx].get("name","?")
        skin_name = gltf["skins"][skin_idx].get("name","?")
        print(f"  node[{i}] '{nd.get('name','?')}' mesh='{mesh_name}' skin='{skin_name}'")

# Materials metallic check
mats = gltf.get("materials",[])
print(f"\n--- Materials ({len(mats)}) ---")
for i, m in enumerate(mats):
    pbr = m.get("pbrMetallicRoughness",{})
    mf = pbr.get("metallicFactor","not set")
    rf = pbr.get("roughnessFactor","not set")
    bc = "yes" if "baseColorTexture" in pbr else "no"
    mr = "yes" if "metallicRoughnessTexture" in pbr else "no"
    nt = "yes" if "normalTexture" in m else "no"
    print(f"  mat[{i}] '{m.get('name','?')}' metallic={mf} roughness={rf} baseColorTex={bc} mrTex={mr} normalTex={nt}")

# Eye position check: compute skinned eye centroid vs head joint
print(f"\n--- Eye position sanity check ---")
# Find eyes skin (fewer joints)
eyes_skin_idx = min(range(len(skins)), key=lambda i: len(skins[i]["joints"]))
body_skin_idx = 1 - eyes_skin_idx if len(skins) == 2 else max(range(len(skins)), key=lambda i: len(skins[i]["joints"]))

es = skins[eyes_skin_idx]
bs = skins[body_skin_idx]
print(f"  Eyes skin: {len(es['joints'])} joints → {[gltf['nodes'][j].get('name') for j in es['joints']]}")

# Get Head joint world position from inverse bind matrix
bs_ibm_idx = bs.get("inverseBindMatrices")
if bs_ibm_idx is not None:
    ibm_data = get_accessor(gltf, bin_data, bs_ibm_idx)  # shape (N, 16)
    bs_joints = bs["joints"]
    bs_jnames = [gltf["nodes"][j].get("name","?") for j in bs_joints]
    head_idx = next((i for i,n in enumerate(bs_jnames) if n.lower()=="head"), None)
    if head_idx is not None:
        ibm = ibm_data[head_idx].reshape(4,4, order='F')  # column-major
        # joint world = inv(IBM)
        head_world = np.linalg.inv(ibm)
        head_world_pos = head_world[:3,3]
        print(f"  Head joint world pos (from body skin IBM): {head_world_pos}")

# Get eyes mesh vertices
# Find node using eyes skin
eyes_mesh_node = next((nd for nd in nodes if nd.get("skin")==eyes_skin_idx and "mesh" in nd), None)
if eyes_mesh_node:
    em_idx = eyes_mesh_node["mesh"]
    em = gltf["meshes"][em_idx]
    pos_idx = em["primitives"][0]["attributes"]["POSITION"]
    pos = get_accessor(gltf, bin_data, pos_idx)
    eye_centroid = pos.mean(axis=0)
    print(f"  Eyes mesh centroid (bind pose local): {eye_centroid}")
    print(f"  Eyes mesh bounds: X[{pos[:,0].min():.3f},{pos[:,0].max():.3f}] Y[{pos[:,1].min():.3f},{pos[:,1].max():.3f}] Z[{pos[:,2].min():.3f},{pos[:,2].max():.3f}]")

    # Read JOINTS_0 and WEIGHTS_0 to skin the centroid
    joints_0_idx = em["primitives"][0]["attributes"].get("JOINTS_0")
    weights_0_idx = em["primitives"][0]["attributes"].get("WEIGHTS_0")
    if joints_0_idx and weights_0_idx:
        j0 = get_accessor(gltf, bin_data, joints_0_idx)
        w0 = get_accessor(gltf, bin_data, weights_0_idx)
        es_ibm_idx = es.get("inverseBindMatrices")
        es_ibm = get_accessor(gltf, bin_data, es_ibm_idx) if es_ibm_idx else None

        if es_ibm is not None and bs_ibm_idx is not None:
            es_joints = es["joints"]
            # Map eye skin joint index → body skin joint index
            eye_to_body = {}
            for local_idx, nidx in enumerate(es_joints):
                if nidx in bs_joints:
                    eye_to_body[local_idx] = bs_joints.index(nidx)
                else:
                    eye_to_body[local_idx] = None

            # Compute skinned eye centroid
            skinned = []
            for vi in range(len(pos)):
                p = np.array([*pos[vi], 1.0], dtype=np.float64)
                sp = np.zeros(3)
                for inf in range(4):
                    ji = int(j0[vi, inf])
                    w = float(w0[vi, inf])
                    if w < 1e-6: continue
                    ei_ibm = es_ibm[ji].reshape(4,4,order='F')
                    # joint world = look up from body skin or compute from node
                    body_ji = eye_to_body.get(ji)
                    if body_ji is not None:
                        bi_ibm = ibm_data[body_ji].reshape(4,4,order='F')
                        jw = np.linalg.inv(bi_ibm)
                    else:
                        jw = np.linalg.inv(ei_ibm)
                    skin_mat = jw @ ei_ibm
                    sp += w * (skin_mat @ p)[:3]
                skinned.append(sp)
            skinned = np.array(skinned)
            sc = skinned.mean(axis=0)
            print(f"  Eyes skinned centroid (bind pose): {sc}")
            if head_idx is not None:
                dist = np.linalg.norm(sc - head_world_pos)
                print(f"  Distance from Head joint: {dist:.4f}  (< 0.15 = OK, > 0.2 = broken)")
                if dist < 0.15:
                    print("  ✓ Eyes appear correctly positioned near head")
                else:
                    print("  ✗ Eyes may be misplaced — check carefully")

print("\nDone.")
