# SWAP Optimization Algorithm Demo

## Problem
Given a set of quantum programs, 20-qubit connectivity graph. Want to find the best SWAP routing strat minimizing the `score = 0.5 * depth + swaps`. Need to implement

```python
def solve(program, hardware_graph) -> (initial_placement: dict, routed_program: list[tuple]): pass
```

Provided kit has a scorer, benchmark programs, con graph, bad baselines.

## Our Solution
Modified SABRE algo: tested a few optimizations on random sample then implemented. (With testing done at n=20.) <10s runtime.

## Setup
### Installs
```bash
python -m pip install networkx
```

### Sample Usage
Notice, the following code requires installing `matplotlib` (for starter_kit).

```
from starter_kit import BENCHMARKS, build_hardware_graph, score_summary
from solver import solve

G = build_hardware_graph()
program = BENCHMARKS["qaoa_random"]
placement, routed = solve(program, G)

s = score_summary(program, G, placement, routed)
print(s["valid"], s["score"], s["swap_count"], s["depth"])
```

Output:
```
True 12.0 6 12
```
