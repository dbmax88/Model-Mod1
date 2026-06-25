#!/usr/bin/env python3
"""Quick visual sanity check: skinned joint positions across frames."""
import numpy as np, math, struct
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pxr import Usd, UsdSkel, UsdGeom, Gf, Vt

USDZ = "giant_pacific_octopus_swim_A3.usdz"
OUT  = "render_check.png"

stage = Usd.Stage.Open(USDZ)
anim  = UsdSkel.Animation(stage.GetPrimAtPath("/World/Octopus/SkelRoot/SkelAnim"))
skel  = UsdSkel.Skeleton(stage.GetPrimAtPath("/World/Octopus/SkelRoot/Skeleton"))

joints = list(skel.GetJointsAttr().Get())
N = len(joints)
print(f"Skeleton: {N} joints")

# Build parent index from joint paths
parent_idx = []
for i, jpath in enumerate(joints):
    parts = jpath.split("/")
    if len(parts) == 1:
        parent_idx.append(-1)
    else:
        parent_path = "/".join(parts[:-1])
        try:
            parent_idx.append(joints.index(parent_path))
        except ValueError:
            parent_idx.append(-1)

def quatf_to_mat(q):
    w,x,y,z = q.GetReal(), q.GetImaginary()[0], q.GetImaginary()[1], q.GetImaginary()[2]
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w),   0],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w),   0],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y), 0],
        [0,              0,              0,              1]
    ])

def frame_joint_positions(tc):
    T = anim.GetTranslationsAttr().Get(tc)
    R = anim.GetRotationsAttr().Get(tc)
    S = anim.GetScalesAttr().Get(tc)
    world = [None]*N
    for i in range(N):
        t = np.array([T[i][0], T[i][1], T[i][2]])
        rm = quatf_to_mat(R[i])
        s = np.array([S[i][0], S[i][1], S[i][2]])
        rm[0,:3] *= s[0]; rm[1,:3] *= s[1]; rm[2,:3] *= s[2]
        local = np.eye(4)
        local[:3,:3] = rm[:3,:3]; local[3,:3] = t
        p = parent_idx[i]
        world[i] = world[p] @ local if p >= 0 else local
    return np.array([w[3,:3] for w in world])

frames = [0, 25, 50, 75, 100]
colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']

fig, axes = plt.subplots(1,2,figsize=(14,7))
fig.suptitle("Octopus skinned joint positions — swim_A3 animation frames 0/25/50/75/100",
             fontsize=11)

for fi, (fr, c) in enumerate(zip(frames, colors)):
    pts = frame_joint_positions(float(fr))
    label = f"frame {fr}"
    for ax, (xi,yi,title) in zip(axes, [(0,1,"Front (X-Y)"),(0,2,"Side (X-Z)")]):
        ax.scatter(pts[:,xi], pts[:,yi], s=3, c=c,
                   alpha=0.5, label=label if ax==axes[0] else None)

for ax, title in zip(axes, ["Front view (X-Y)", "Side view (X-Z)"]):
    ax.set_title(title); ax.set_aspect("equal")
    ax.grid(True, alpha=0.3); ax.set_xlabel("X"); ax.set_ylabel(title.split("-")[1][0])
axes[0].legend(markerscale=5, loc="upper right")

plt.tight_layout()
plt.savefig(OUT, dpi=130, bbox_inches="tight")
print(f"Saved → {OUT}")
plt.close()

# Print Y range per frame to confirm above-ground and swimming motion
print("\nJoint Y range per frame (scaled coords):")
for fr in frames:
    pts = frame_joint_positions(float(fr))
    print(f"  frame {fr:3d}: Y=[{pts[:,1].min():.4f}, {pts[:,1].max():.4f}]  "
          f"X=[{pts[:,0].min():.4f}, {pts[:,0].max():.4f}]")
