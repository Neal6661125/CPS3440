# Advanced K-Shortest Paths Solver

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A realistic path-finding engine that finds the **top-K shortest paths** in a grid-like urban network, supporting:

- Time-dependent edge weights (rush hour, lunch time, late night)
- Turning/geometric penalties at intersections (discourages sharp turns)
- Stochastic edge noise (simulates real-time traffic fluctuation)
- Euclidean-distance heuristic A*
- Pruned recursive enumeration for K-shortest loopless paths



## Features

| Feature                     | Description                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| Time-of-day cost multiplier | ×1.4 peak (17–19h), ×1.3 morning rush, ×0.8 late night, etc.                |
| Turning penalty (λ)         | Penalty based on actual turning angle (0° → 180°), up to +100%+ extra cost |
| Edge weight noise           | Random jitter ±variance on every query to mimic live traffic               |
| A* with Euclidean heuristic | Dramatically reduces expanded nodes                                        |
| Pruned K-path search        | Uses 1.5× best-path cost as hard bound → stays fast even for K=20+         |
| Path reliability metric    | Sum of variances along the path (lower = more stable/predictable)          |

## Network Overview

Built-in 6×5 grid + extra node Z (26 nodes total, ~340 bidirectional edges):

- Horizontal/vertical edges: base cost 10
- Diagonal edges (45°/135°): base cost ≈14.14
- All edges have configurable variance and direction
