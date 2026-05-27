# sim_teleop

Remote teleoperation of robotic arms in simulation, powered by the [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) physics engine.

## Goals

1. **Sim teleop for real robot arms** -- build a simulation environment where you can remotely teleoperate the same arms we use on the bench (SO100, Aloha, Koch), without needing the physical hardware connected.

2. **Try the Genesis engine** -- Genesis is a Python-first physics simulator that runs 10-80x faster than Isaac Sim and MuJoCo. We want to evaluate it as our primary sim backend for policy training and teleop data collection.

3. **Sim-to-real pipeline** -- collect teleop demonstrations in sim, train policies, and transfer to real hardware via the existing LeRobot / GR00T infrastructure in BluPe.

## Supported Arms

| Arm | DOF | Format | Status |
|-----|-----|--------|--------|
| SO100 (Feetech) | 6 | URDF | Planned |
| Aloha | 14 (dual) | MJCF | Planned |
| Koch | 6 | URDF | Planned |

## Architecture

```
┌─────────────┐      ┌──────────────────┐      ┌─────────────┐
│  Teleop      │      │  Genesis Scene   │      │  Data        │
│  Interface   │─────▶│  (physics sim)   │─────▶│  Recording   │
│  (gamepad,   │      │                  │      │  (LeRobot    │
│   keyboard,  │      │  Robot + Objects  │      │   format)    │
│   phone)     │      │  + Cameras       │      │              │
└─────────────┘      └──────────────────┘      └─────────────┘
```

- **Teleop Interface**: gamepad, keyboard, or phone input mapped to joint targets
- **Genesis Scene**: physics simulation with robot arm, table, objects, and cameras
- **Data Recording**: save episodes in LeRobot-compatible format for downstream training

## Getting Started

```bash
# Clone
git clone https://github.com/andlyu/sim_teleop.git
cd sim_teleop

# Install dependencies
pip install genesis-world
pip install -e .
```

> Full setup instructions coming soon.

## Roadmap

- [ ] Project scaffolding and Genesis "hello world"
- [ ] Load SO100 URDF into Genesis scene
- [ ] Keyboard teleop in sim
- [ ] Camera rendering and observation recording
- [ ] LeRobot-format episode export
- [ ] Gamepad / phone teleop support
- [ ] Multi-arm (Aloha) setup
- [ ] Sim-to-real transfer experiments

## Related

- [BluPe](https://github.com/andlyu) -- parent project (real robot teleop, GR00T integration)
- [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) -- physics engine
- [LeRobot](https://github.com/huggingface/lerobot) -- robot learning framework
