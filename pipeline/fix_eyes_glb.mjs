#!/usr/bin/env node
/**
 * fix_eyes_glb.mjs
 *
 * Reads Hee_Swim_fixed.glb (metallic fix already applied) and:
 *  1. Repositions the eyes mesh vertices: right/left clusters recentered,
 *     scaled to 35% size, placed at sensible positions on the head.
 *  2. Rebinds every eye vertex to the Head joint only (weight 1,0,0,0).
 *  3. Adds Head to the eyes skin's joint list with the correct IBM.
 *
 * Formula matches fix_eyes_placement.py exactly.
 * Writes Hee_Swim_final.glb.
 *
 * API note: Accessor.getCount() derives from array.length / elementSize —
 * there is no setCount(); setArray() with the correctly-sized array is all
 * that's needed.
 */

import { NodeIO } from '@gltf-transform/core';
import { KHRMeshQuantization } from '@gltf-transform/extensions';

// ── Constants matching Python fix (fix_eyes_placement.py) ────────────────────
const EYE_SCALE = 0.35;
const CENTER_R = [+0.085, 0.12, 0.30]; // right eye new center, head local space
const CENTER_L = [-0.085, 0.12, 0.30]; // left eye new center, head local space

// componentType constants
const SIGNED_SHORT = 5122;  // Int16Array
const FLOAT = 5126;
const UNSIGNED_BYTE = 5121;
const UNSIGNED_SHORT = 5123;

// ── Helpers ───────────────────────────────────────────────────────────────────

function decodeNormInt16(v) {
  return Math.max(v / 32767, -1.0);
}

function encodeNormInt16(f) {
  return Math.round(Math.max(-1, Math.min(1, f)) * 32767);
}

function centroid3(floatXYZ, indices) {
  let cx = 0, cy = 0, cz = 0;
  for (const i of indices) {
    cx += floatXYZ[i * 3];
    cy += floatXYZ[i * 3 + 1];
    cz += floatXYZ[i * 3 + 2];
  }
  const n = indices.length;
  return [cx / n, cy / n, cz / n];
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function main() {
  const io = new NodeIO().registerExtensions([KHRMeshQuantization]);

  console.log('Reading Hee_Swim_fixed.glb …');
  const doc = await io.read('Hee_Swim_fixed.glb');
  const root = doc.getRoot();

  // ── Find meshes ────────────────────────────────────────────────────────────
  const meshes = root.listMeshes();
  console.log('Meshes:', meshes.map(m => m.getName()));

  let eyesMesh = null;
  for (const mesh of meshes) {
    if (mesh.getName().toLowerCase().includes('eye')) {
      eyesMesh = mesh;
      break;
    }
  }
  if (!eyesMesh) {
    // Fallback: pick the mesh whose skin has the fewest joints
    const skins = root.listSkins();
    const eyesSkinFallback = skins.reduce((a, b) =>
      a.listJoints().length < b.listJoints().length ? a : b
    );
    for (const node of root.listNodes()) {
      if (node.getSkin() === eyesSkinFallback && node.getMesh()) {
        eyesMesh = node.getMesh();
        break;
      }
    }
  }
  if (!eyesMesh) throw new Error('Cannot find eyes mesh');
  console.log('Eyes mesh:', eyesMesh.getName());

  // ── Find skins ─────────────────────────────────────────────────────────────
  const skins = root.listSkins();
  console.log('Skins:', skins.map(s => `${s.getName()} (${s.listJoints().length} joints)`));

  let eyesSkin = null, bodySkin = null;
  for (const skin of skins) {
    if (skin.listJoints().length <= 10) eyesSkin = skin;
    else bodySkin = skin;
  }
  if (!eyesSkin || !bodySkin) throw new Error(
    `Expected one small skin and one large skin, got: ${skins.map(s => s.listJoints().length)}`
  );
  console.log(`Eyes skin: ${eyesSkin.listJoints().length} joints`);
  console.log(`Body skin: ${bodySkin.listJoints().length} joints`);

  // ── Find Head joint in body skin ───────────────────────────────────────────
  const bodyJoints = bodySkin.listJoints();
  let headJointIdxInBody = -1;
  let headNode = null;

  for (let i = 0; i < bodyJoints.length; i++) {
    const n = bodyJoints[i].getName();
    if (n === 'Head' || n.toLowerCase() === 'head') {
      headJointIdxInBody = i;
      headNode = bodyJoints[i];
      break;
    }
  }
  // Fuzzy fallback
  if (!headNode) {
    for (let i = 0; i < bodyJoints.length; i++) {
      if (bodyJoints[i].getName().toLowerCase().includes('head')) {
        headJointIdxInBody = i;
        headNode = bodyJoints[i];
        break;
      }
    }
  }
  if (!headNode) throw new Error(
    'Cannot find Head joint in body skin. Joints sample: ' +
    bodyJoints.slice(0, 20).map(j => j.getName()).join(', ')
  );
  console.log(`Head joint: "${headNode.getName()}" at body skin index ${headJointIdxInBody}`);

  // ── Copy Head's IBM from body skin → eyes skin ─────────────────────────────
  const bodyIbmAcc = bodySkin.getInverseBindMatrices();
  if (!bodyIbmAcc) throw new Error('Body skin has no IBM accessor');
  const bodyIbmArray = bodyIbmAcc.getArray(); // Float32Array, 16 floats per joint
  const headIbm = bodyIbmArray.slice(headJointIdxInBody * 16, headJointIdxInBody * 16 + 16);
  console.log('Head IBM diagonal (should be non-zero):', headIbm[0].toFixed(4), headIbm[5].toFixed(4), headIbm[10].toFixed(4), headIbm[15].toFixed(4));

  const eyesIbmAcc = eyesSkin.getInverseBindMatrices();
  if (!eyesIbmAcc) throw new Error('Eyes skin has no IBM accessor');

  const oldEyesIbm = eyesIbmAcc.getArray();
  const newEyesIbm = new Float32Array(oldEyesIbm.length + 16);
  newEyesIbm.set(oldEyesIbm);
  newEyesIbm.set(headIbm, oldEyesIbm.length);

  // setArray() is all we need — getCount() is derived from array.length / elementSize,
  // there is no setCount() in gltf-transform v4.
  eyesIbmAcc.setArray(newEyesIbm);
  console.log(`Eyes IBM: ${oldEyesIbm.length / 16} → ${newEyesIbm.length / 16} entries (count auto-derives from array length)`);

  // Head's index in eyes skin = current joint count (before addJoint)
  const headJointIdxInEyes = eyesSkin.listJoints().length; // == 4 (was 0..3)
  eyesSkin.addJoint(headNode);
  console.log(`Head added to eyes skin at index ${headJointIdxInEyes}`);
  console.log(`Eyes skin now has ${eyesSkin.listJoints().length} joints`);

  // ── Fix the eyes mesh POSITION ─────────────────────────────────────────────
  const eyesPrim = eyesMesh.listPrimitives()[0];
  const posAcc = eyesPrim.getAttribute('POSITION');
  const posComponentType = posAcc.getComponentType();
  const posNormalized = posAcc.getNormalized();
  const vertCount = posAcc.getCount();

  console.log(`POSITION: componentType=${posComponentType}, normalized=${posNormalized}, count=${vertCount}`);

  // Read raw array and decode to float
  const rawPos = posAcc.getArray();
  const floatPos = new Float32Array(vertCount * 3);

  if (posNormalized && posComponentType === SIGNED_SHORT) {
    for (let i = 0; i < rawPos.length; i++) floatPos[i] = decodeNormInt16(rawPos[i]);
  } else if (posComponentType === FLOAT) {
    floatPos.set(rawPos);
  } else {
    // Use getElement for any other encoding
    const tmp = [0, 0, 0];
    for (let i = 0; i < vertCount; i++) {
      posAcc.getElement(i, tmp);
      floatPos[i * 3] = tmp[0];
      floatPos[i * 3 + 1] = tmp[1];
      floatPos[i * 3 + 2] = tmp[2];
    }
  }

  // Split into right (X > 0) and left (X ≤ 0) clusters
  const rightIdx = [], leftIdx = [];
  for (let i = 0; i < vertCount; i++) {
    if (floatPos[i * 3] > 0) rightIdx.push(i);
    else leftIdx.push(i);
  }
  console.log(`Clusters: right=${rightIdx.length} verts, left=${leftIdx.length} verts`);

  const cR = centroid3(floatPos, rightIdx);
  const cL = centroid3(floatPos, leftIdx);
  console.log('Right cluster centroid (pre-fix):', cR.map(v => v.toFixed(4)));
  console.log('Left  cluster centroid (pre-fix):', cL.map(v => v.toFixed(4)));

  // Reposition: new_pos = (old_pos - cluster_center) * EYE_SCALE + new_center
  const newFloatPos = new Float32Array(vertCount * 3);
  for (const i of rightIdx) {
    newFloatPos[i * 3]     = (floatPos[i * 3]     - cR[0]) * EYE_SCALE + CENTER_R[0];
    newFloatPos[i * 3 + 1] = (floatPos[i * 3 + 1] - cR[1]) * EYE_SCALE + CENTER_R[1];
    newFloatPos[i * 3 + 2] = (floatPos[i * 3 + 2] - cR[2]) * EYE_SCALE + CENTER_R[2];
  }
  for (const i of leftIdx) {
    newFloatPos[i * 3]     = (floatPos[i * 3]     - cL[0]) * EYE_SCALE + CENTER_L[0];
    newFloatPos[i * 3 + 1] = (floatPos[i * 3 + 1] - cL[1]) * EYE_SCALE + CENTER_L[1];
    newFloatPos[i * 3 + 2] = (floatPos[i * 3 + 2] - cL[2]) * EYE_SCALE + CENTER_L[2];
  }

  // Verify new centroids
  const newCR = centroid3(newFloatPos, rightIdx);
  const newCL = centroid3(newFloatPos, leftIdx);
  console.log('Right cluster centroid (post-fix):', newCR.map(v => v.toFixed(4)), '  expected:', CENTER_R);
  console.log('Left  cluster centroid (post-fix):', newCL.map(v => v.toFixed(4)), '  expected:', CENTER_L);

  // Requantize back to the same encoding
  if (posNormalized && posComponentType === SIGNED_SHORT) {
    const newRawPos = new Int16Array(vertCount * 3);
    for (let i = 0; i < newRawPos.length; i++) newRawPos[i] = encodeNormInt16(newFloatPos[i]);
    posAcc.setArray(newRawPos);
  } else if (posComponentType === FLOAT) {
    posAcc.setArray(newFloatPos);
  } else {
    const tmp = [0, 0, 0];
    for (let i = 0; i < vertCount; i++) {
      tmp[0] = newFloatPos[i * 3]; tmp[1] = newFloatPos[i * 3 + 1]; tmp[2] = newFloatPos[i * 3 + 2];
      posAcc.setElement(i, tmp);
    }
  }
  console.log('POSITION fixed.');

  // ── Fix JOINTS_0 — all vertices → Head joint only ─────────────────────────
  const jointsAcc = eyesPrim.getAttribute('JOINTS_0');
  if (!jointsAcc) throw new Error('Eyes prim has no JOINTS_0');
  const jointsArr = jointsAcc.getArray();
  const newJointsArr = new (jointsArr.constructor)(vertCount * 4);
  // All zeros by default; set first influence to headJointIdxInEyes
  for (let i = 0; i < vertCount; i++) {
    newJointsArr[i * 4] = headJointIdxInEyes; // index 4, fits in uint8
    // [1],[2],[3] stay 0
  }
  jointsAcc.setArray(newJointsArr);
  console.log(`JOINTS_0 fixed — all vertices → joint ${headJointIdxInEyes} (Head).`);

  // ── Fix WEIGHTS_0 — weight [1, 0, 0, 0] ───────────────────────────────────
  const weightsAcc = eyesPrim.getAttribute('WEIGHTS_0');
  if (!weightsAcc) throw new Error('Eyes prim has no WEIGHTS_0');
  const weightsComponentType = weightsAcc.getComponentType();
  const weightsNormalized = weightsAcc.getNormalized();

  let newWeightsArr;
  if (weightsComponentType === FLOAT) {
    newWeightsArr = new Float32Array(vertCount * 4);
    for (let i = 0; i < vertCount; i++) newWeightsArr[i * 4] = 1.0;
  } else if (weightsComponentType === UNSIGNED_SHORT && weightsNormalized) {
    newWeightsArr = new Uint16Array(vertCount * 4);
    for (let i = 0; i < vertCount; i++) newWeightsArr[i * 4] = 65535;
  } else {
    // UNSIGNED_BYTE normalized (most common in quantized assets)
    newWeightsArr = new Uint8Array(vertCount * 4);
    for (let i = 0; i < vertCount; i++) newWeightsArr[i * 4] = 255;
  }
  weightsAcc.setArray(newWeightsArr);
  console.log(`WEIGHTS_0 fixed (componentType=${weightsComponentType}, normalized=${weightsNormalized}).`);

  // ── Write output ───────────────────────────────────────────────────────────
  console.log('Writing Hee_Swim_final.glb …');
  await io.write('Hee_Swim_final.glb', doc);
  console.log('Done → Hee_Swim_final.glb');
}

main().catch(err => { console.error(err); process.exit(1); });
