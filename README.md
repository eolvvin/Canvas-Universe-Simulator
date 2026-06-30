# Canvas Universe Simulator

A real-time 3D simulation of the **Canvas Model** — a unified framework for fundamental physics where spacetime, particles, and gravity emerge from wave intersections on a pre-geometric canvas.

## What This Simulates

The Canvas Model proposes that the universe is built from eight primitives and four equations. This simulation demonstrates the core mechanism:

- **Oscillators** generate continuous space and time waves that propagate across a 3D grid
- **Threshold crossing** — when space and time wave amplitudes exceed a critical value, a **spacetime voxel** forms
- **Particle creation** — new voxels can spawn particles that move under emergent gravity
- **The Unified Wave Equation** governs all field dynamics with a nonlinear Polarity term

## What You'll See

| Visual | Meaning |
|--------|---------|
| Blue glow | Wave field intensity |
| Warm golden glow | Spacetime voxels (brighter = older) |
| Yellow dots | Particles moving under gravity |

## Controls

| Key | Action |
|-----|--------|
| `Q` / `E` | Zoom in / out |
| Arrow keys | Pan the view |
| `A` / `S` / `D` | Switch to YZ / XZ / XY slice |
| `W` / `X` | Move slice forward / back |
| `1` / `2` | Combined fields / Time field view |
| `R` | Reset simulation |
| `Space` | Pause / resume |

## Physics Parameters

The simulation uses the derived Canvas Model parameters:

- `c_eff = 0.00446` — Effective Acceleration weight
- `d_eff = 0.00284` — Effective Polarity weight
- `c_eff / d_eff = π/2 ≈ 1.5708` — The central falsifiable prediction
- `T_ST = 4.0` — Spacetime threshold
- `σ = 0.5` — Higgs width from threshold condition

## Requirements

- Python 3.8+
- [Taichi](https://github.com/taichi-dev/taichi) — GPU-accelerated computing
- A CUDA-compatible GPU (or CPU fallback)

## Installation

```bash
pip install taichi
