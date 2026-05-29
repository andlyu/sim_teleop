# sim_teleop — Plan

High-level roadmap for the SO101-in-Genesis teleop + policy sandbox.
This is the canonical reference; details get filled in per subtask.

## North star

An **SO101 arm in a realistic Genesis simulation**, controllable by the
**MolmoAct2** policy, that a human can **reset / intervene on** by hand.

The deeper purpose: a sandbox to run (and eventually improve) a real VLA
policy on our robot **without the physical hardware in the loop** — record,
evaluate, and iterate in sim, with a path toward sim-to-real.

## What we already know (the contracts)

- **Sim engine:** Genesis (`genesis-world==1.0.0`, `torch==2.12.0`), verified
  running on Apple Silicon (CPU/Metal). Core loop: `init -> add_entity ->
  build -> step`. Robot loads natively via `gs.morphs.URDF`.
- **Robot:** SO101 (SO-ARM100 family), 6 revolute joints + gripper. URDF +
  meshes vendored under `assets/so101/`. Joint order:
  `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`.
- **Policy:** `allenai/MolmoAct2-SO100_101` — a 5B VLA covering the SO100/SO101
  family (norm_tag `so100_so101_molmoact2`), so SO101 is fully supported.
  - **Inputs per step:** 2 RGB images (top + side; order-agnostic),
    6-D proprioceptive state **in raw robot scale** (degrees, NOT radians),
    a natural-language instruction.
  - **Output:** continuous actions `(N, D)` in robot scale, un-normalized via
    `norm_tag="so100_so101_molmoact2"`. Depth reasoning disabled for this ckpt.
  - **Runs on:** a GPU (bf16 < 16GB -> cheap Vast box, e.g. RTX 3090/4090).
    Plain `transformers` + `trust_remote_code`, ~180ms/action. NOT the Mac.

## Design principle

Subtasks 3-4 build a **policy-agnostic harness**: teleop, a stub policy, and
MolmoAct all look identical to the sim — `observation -> action`. That seam is
what makes "swap the human for the net" a near-trivial change, and it lets us
debug the whole loop on the Mac (free) before renting a GPU (paid).

## Subtasks

Subtasks **1, 2, and 3 can be developed in parallel** (each against stubs);
they converge for integration. Subtask **4 integrates** and comes after.

### 1. Creating the setup
The Genesis environment: SO101 + table + a task object, loaded and
controllable (`control_dofs_position`). A gym-style `reset()` / `step(action)`
/ `get_observation()` interface. Headless is fine to start — proving the
physics/control core, not looks. (Realism pass — lighting, ray tracing,
rope/water — layers on here later, once the loop works.)

### 2. Placing cameras
Two external RGB views (**top + side**) matching MolmoAct2's expected input.
This is what turns the scene into *observations*. Pin resolution to the
checkpoint's spec. Develops in parallel against the setup stub.

### 3. Adding the policy
The `observation -> action` interface, plus:
- a **stub policy** (scripted/random actions, right shape) for free Mac-side
  testing,
- the **state/action adapter** (Genesis radians <-> SO100 raw scale) — the #1
  silent-failure point, owns its own attention,
- the **MolmoAct2 client** (talks to the remote GPU server),
- the **Vast GPU server** hosting `MolmoAct2-SO100_101` (stand up last; rent
  only once the Mac side works end-to-end with the stub).

### 4. Integrating remote teleop
Keyboard / remote teleop to **reset the scene** between rollouts (and
optionally **take over** when the policy fails). Reuses the same control path
as subtask 1. Closes the loop: policy acts -> human resets -> policy acts again.

## Phases

- **Phase A — Mac (free):** subtasks 1-3 (with stub) + 4 -> a working loop
  driven by the stub policy and teleop.
- **Phase B — Vast (paid):** swap the stub for `MolmoAct2-SO100_101` on a
  rented GPU. Mostly a config + network-seam change if Phase A is solid.
- **Phase C — Polish:** realism (lighting/ray tracing), then the "cool demo"
  (rope / water manipulation via Genesis's deformable + fluid solvers).

## Resolved decisions

- **SO100 vs SO101 → SO101 is canonical** (2026-05-29). The MolmoAct2-SO100_101
  checkpoint covers both arms, and the SO101 model (URDF + 13 meshes) is already
  vendored and verified loading/teleoping in Genesis. The stale `assets/so100/`
  (meshless URDF) is superseded by `assets/so101/`.
- **Realism is Phase C (last).** Build the observation→action core loop first;
  table/lighting/ray-tracing polish layers on once the loop works.

## Open questions

- Manipulation target for the first working version: rigid object (recommended
  through Phase A) vs. rope/water (Phase C).
- Sim-only demo vs. stepping stone to the real SO100 (raises the visual-realism
  bar — the same model must work on sim and real cameras).
- Vast access method (CLI vs. web) and persistent-volume strategy for the
  checkpoint.
