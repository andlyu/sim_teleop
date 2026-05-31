# SO101 model assets

These files (`so101_new_calib.urdf`, `so101_old_calib.urdf`, and `assets/*.stl`)
are vendored from:

- **TheRobotStudio/SO-ARM100** — https://github.com/TheRobotStudio/SO-ARM100
- Path upstream: `Simulation/SO101/`
- License: **Apache-2.0** (see upstream `LICENSE`)

`so101_new_calib.urdf` uses the **new calibration** (each joint's virtual zero is
the middle of its range). `so101_old_calib.urdf` uses the older convention where
zero corresponds to the fully extended horizontal arm. The model is 6-DOF; joint
names: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`,
`wrist_roll`, `gripper`.

Note (from upstream): in LeRobot the gripper is a linear 0–100 joint, but here
it is modeled as revolute. Base collision meshes were removed upstream due to
problematic collision behavior.
