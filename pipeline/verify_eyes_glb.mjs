#!/usr/bin/env node
/**
 * verify_eyes_glb.mjs
 *
 * Skinning check: samples a handful of eye vertices from Hee_Swim_final.glb,
 * applies the corrected skin matrix (IBM × inverse-node-world × joint-world),
 * and confirms the resulting world positions land close to the Head joint's
 * world-space location — not 0.2+ units away as they were before the fix.
 *
 * Uses only bind-pose geometry (animation frame 0 / rest pose) since we only
 * need to verify the IBM/joint relationship is sane.
 */

import { NodeIO } from '@gltf-transform/core';
import { KHRMeshQuantization } from '@gltf-transform/extensions';

// 4×4 matrix helpers (column-major, matching glTF / WebGL convention)
function mat4Identity() {
  return new Float64Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]);
}

function mat4FromTRS(t, r, s) {
  // r is [x,y,z,w] quaternion
  const [x, y, z, w] = r;
  const [sx, sy, sz] = s;
  const [tx, ty, tz] = t;
  return new Float64Array([
    (1 - 2*(y*y + z*z)) * sx,  (2*(x*y + z*w)) * sx,     (2*(x*z - y*w)) * sx,     0,
    (2*(x*y - z*w)) * sy,      (1 - 2*(x*x + z*z)) * sy,  (2*(y*z + x*w)) * sy,     0,
    (2*(x*z + y*w)) * sz,      (2*(y*z - x*w)) * sz,      (1 - 2*(x*x + y*y)) * sz, 0,
    tx, ty, tz, 1
  ]);
}

function mat4Mul(A, B) {
  const C = new Float64Array(16);
  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 4; col++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) sum += A[row + k * 4] * B[k + col * 4];
      C[row + col * 4] = sum;
    }
  }
  return C;
}

// Transform a point [x,y,z] by a 4×4 matrix (column-major)
function transformPoint(M, p) {
  const x = M[0]*p[0] + M[4]*p[1] + M[8]*p[2]  + M[12];
  const y = M[1]*p[0] + M[5]*p[1] + M[9]*p[2]  + M[13];
  const z = M[2]*p[0] + M[6]*p[1] + M[10]*p[2] + M[14];
  const w = M[3]*p[0] + M[7]*p[1] + M[11]*p[2] + M[15];
  return [x/w, y/w, z/w];
}

// Compute node's local matrix from TRS / matrix property
function nodeLocalMatrix(node) {
  if (node.getMatrix()) return new Float64Array(node.getMatrix());
  const t = node.getTranslation() || [0, 0, 0];
  const r = node.getRotation()    || [0, 0, 0, 1];
  const s = node.getScale()       || [1, 1, 1];
  return mat4FromTRS(t, r, s);
}

// Walk up the scene graph to get world matrix
const worldMatrixCache = new Map();
function nodeWorldMatrix(node) {
  if (worldMatrixCache.has(node)) return worldMatrixCache.get(node);
  const local = nodeLocalMatrix(node);
  const parent = node.getParentNode ? node.getParentNode() : null;
  const result = parent ? mat4Mul(nodeWorldMatrix(parent), local) : local;
  worldMatrixCache.set(node, result);
  return result;
}

// Invert a 4×4 matrix (general, via Gauss-Jordan)
function mat4Inv(m) {
  const inv = new Float64Array(16);
  const src = Array.from(m);
  // augmented [src | identity]
  const aug = [];
  for (let i = 0; i < 4; i++) {
    aug.push([...src.slice(i*4, i*4+4), ...[0,0,0,0].map((_, j) => j === i ? 1 : 0)]);
  }
  // Column-major → row-major for Gauss-Jordan
  const mat = [];
  for (let row = 0; row < 4; row++) {
    const r = [];
    for (let col = 0; col < 4; col++) r.push(m[col * 4 + row]);
    for (let col = 0; col < 4; col++) r.push(col === row ? 1 : 0);
    mat.push(r);
  }
  for (let col = 0; col < 4; col++) {
    let pivotRow = col;
    for (let r = col+1; r < 4; r++) if (Math.abs(mat[r][col]) > Math.abs(mat[pivotRow][col])) pivotRow = r;
    [mat[col], mat[pivotRow]] = [mat[pivotRow], mat[col]];
    const piv = mat[col][col];
    if (Math.abs(piv) < 1e-12) throw new Error('Singular matrix');
    for (let j = 0; j < 8; j++) mat[col][j] /= piv;
    for (let r = 0; r < 4; r++) {
      if (r === col) continue;
      const f = mat[r][col];
      for (let j = 0; j < 8; j++) mat[r][j] -= f * mat[col][j];
    }
  }
  // Extract result (row-major result → column-major)
  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 4; col++) inv[col * 4 + row] = mat[row][col + 4];
  }
  return inv;
}

function dist3(a, b) {
  return Math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2);
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function main() {
  const io = new NodeIO().registerExtensions([KHRMeshQuantization]);
  const doc = await io.read('Hee_Swim_final.glb');
  const root = doc.getRoot();

  // Find eyes mesh node (node that has a mesh and uses the small skin)
  const skins = root.listSkins();
  const eyesSkin = skins.reduce((a, b) => a.listJoints().length < b.listJoints().length ? a : b);
  const eyesJoints = eyesSkin.listJoints();
  const eyesIbmAcc = eyesSkin.getInverseBindMatrices();
  const eyesIbm = eyesIbmAcc.getArray();

  console.log(`Eyes skin: ${eyesJoints.length} joints`);
  console.log('Joints:', eyesJoints.map(j => j.getName()));

  // Find Head joint (should be last, index 4)
  const headJoint = eyesJoints[eyesJoints.length - 1];
  console.log(`Head joint in eyes skin: "${headJoint.getName()}" at index ${eyesJoints.length - 1}`);

  // Get eyes mesh node
  let eyesMeshNode = null;
  for (const node of root.listNodes()) {
    if (node.getSkin() === eyesSkin && node.getMesh()) { eyesMeshNode = node; break; }
  }
  if (!eyesMeshNode) throw new Error('Cannot find eyes mesh node');

  const eyesMesh = eyesMeshNode.getMesh();
  const eyesPrim = eyesMesh.listPrimitives()[0];
  const posAcc = eyesPrim.getAttribute('POSITION');
  const jointsAcc = eyesPrim.getAttribute('JOINTS_0');
  const weightsAcc = eyesPrim.getAttribute('WEIGHTS_0');
  const vertCount = posAcc.getCount();

  // Check that all vertices use the Head joint (index should be eyesJoints.length - 1)
  const headIdx = eyesJoints.length - 1;
  let badJointCount = 0;
  const jointEl = [0, 0, 0, 0];
  for (let i = 0; i < vertCount; i++) {
    jointsAcc.getElement(i, jointEl);
    if (jointEl[0] !== headIdx) badJointCount++;
  }
  console.log(`\nJOINTS_0 check: ${vertCount - badJointCount}/${vertCount} verts use Head joint (index ${headIdx}). Bad: ${badJointCount}`);

  // Compute skinned positions for a sample of vertices
  // skinMatrix = joint_world_matrix * IBM
  // final_pos = skinMatrix * mesh_node_inverse_world * bind_pos
  //           = joint_world * IBM * bind_pos   (if mesh node is identity or we work in world space)

  // Get IBM for Head (last entry in eyes IBM)
  const headIbm = eyesIbm.slice(headIdx * 16, headIdx * 16 + 16);

  // World matrices
  const headWorldMat = nodeWorldMatrix(headJoint);
  const meshWorldMat = nodeWorldMatrix(eyesMeshNode);
  const meshWorldInv = mat4Inv(meshWorldMat);

  // Skin matrix: transforms from mesh local space → world space
  const skinMat = mat4Mul(headWorldMat, new Float64Array(headIbm));
  // In glTF, skinned_pos_local = meshWorldInv * skinMat * bind_pos
  const finalMat = mat4Mul(meshWorldInv, skinMat);

  // Sample 10 vertices (evenly spaced)
  const step = Math.max(1, Math.floor(vertCount / 10));
  const samplePos = [0, 0, 0];
  console.log('\nSample skinned positions (should be near Head joint world pos):');

  // Head joint world position
  const headWorldPos = [headWorldMat[12], headWorldMat[13], headWorldMat[14]];
  console.log(`Head joint world pos: [${headWorldPos.map(v => v.toFixed(4)).join(', ')}]`);

  let maxDist = 0;
  for (let i = 0; i < vertCount; i += step) {
    posAcc.getElement(i, samplePos);
    const skinned = transformPoint(finalMat, samplePos);
    const d = dist3(skinned, headWorldPos);
    if (d > maxDist) maxDist = d;
    console.log(`  vert[${i}]: bind=[${samplePos.map(v => v.toFixed(3)).join(', ')}] → skinned=[${skinned.map(v => v.toFixed(3)).join(', ')}]  dist_from_head=${d.toFixed(4)}`);
  }

  console.log(`\nMax distance from Head joint across sampled vertices: ${maxDist.toFixed(4)}`);
  console.log(`Expected: < ~0.15 (head mesh radius)  Previously broken: ~0.22–0.24`);

  if (maxDist < 0.20) {
    console.log('PASS: Eye vertices are within the expected range of the Head joint.');
  } else {
    console.log('WARN: Eye vertices may still be too far from the Head joint. Check the fix.');
  }
}

main().catch(err => { console.error(err); process.exit(1); });
