# Mini Emergency Exit Planner - Technical Report

**Course topic:** 05 - Mini Emergency Exit Planner  
**Author:** [Enter student name]  
**Date:** 19 November 2026

## 1. Problem and objective

A shortest route can become unsafe when a corridor is blocked. This classroom simulation combines search, Bayesian inference, and a one-step Markov decision model: A* plans to Exit A; a noisy smoke reading updates the blockage belief; the MDP compares proceeding and detouring; and a detour closes the monitored cell before A* plans to Exit B.

This is a mathematical demonstration, not a life-safety product. The map, sensor likelihoods, and rewards are hypothetical. There are no real sensor feeds, building drawings, fire models, or human movement dynamics.

## 2. Environment and assumptions

The baseline is a 6 x 6 grid with four-directional moves of cost 1. Start is `(2, 0)`, Exit A `(2, 5)`, Exit B `(5, 5)`. The three fixed walls are `(1, 1)`, `(3, 1)`, and `(4, 3)`. The monitored corridor is `(2, 3)`. Rows and columns are zero-indexed. The stress grid is 10 x 10 with fixed additional walls, defined in `config.py`.

Prior blockage probability is 0.25. Smoke sensor rates are P(Alert|Blocked)=0.85 and P(Alert|Clear)=0.15. Optional heat sensor rates are 0.80 and 0.20 and are assumed conditionally independent of the smoke sensor given the corridor state.

Rewards are +100 for proceeding when clear, -100 for proceeding when blocked, and +40 for detouring. The interface allows the blocked-state penalty to vary; other rewards remain fixed.

## 3. Architecture and algorithm integration

`algorithms/astar.py` provides A* and BFS using the same four neighbors and map. A* orders nodes by `f(n)=g(n)+h(n)`, where `h` is Manhattan distance. Unit step costs and four-way movement make this heuristic admissible and consistent. `algorithms/bayes.py` computes the likelihood and evidence for each binary observation. `algorithms/mdp.py` computes one-step expected utilities and selects the larger. `pipeline.py` executes the closed loop and records both the initial and selected routes.

When DETOUR is selected, the monitored corridor is added to the blocked cells and a new A* search targets Exit B. If no route exists, the result is `path=None`; the UI reports the unreachable exit instead of attempting playback.

## 4. Verified calculations

Bayes' rule is `P(B|E)=P(E|B)P(B)/[P(E|B)P(B)+P(E|not B)P(not B)]`.

- Clear: evidence is `0.15*0.25 + 0.85*0.75 = 0.675`; posterior is `0.0375/0.675 = 0.05556` (5.56%).
- Alert: evidence is `0.85*0.25 + 0.15*0.75 = 0.325`; posterior is `0.2125/0.325 = 0.65385` (65.38%).

At the default penalty, `EU(Proceed)=(1-p)*100+p*(-100)=100-200p`; `EU(Detour)=40`. Equality gives `p*=0.30`. Clear therefore proceeds (EU 88.89 vs 40). Alert detours (EU -30.77 vs 40). At exactly 30%, ties choose DETOUR.

## 5. Testing and benchmark

The automated suite contains nine tests covering both baseline sensor cases, a noisy sensor crossing the boundary, sequential heat evidence, prior extremes, exact MDP tie behavior, unreachable/blocked destinations, invalid probabilities, and A*/BFS cost agreement on both maps. It runs under `python -m pytest -q`; the same tests also run with Python's standard-library unittest runner.

One local run (Python 3.12.14; milliseconds vary by host):

| Map | Search | Path cost | Expanded nodes | Runtime (ms) |
|---|---:|---:|---:|---:|
| 6 x 6 baseline | A* | 5 | 6 | 0.0153 |
| 6 x 6 baseline | BFS | 5 | 22 | 0.0405 |
| 10 x 10 stress | A* | 9 | 10 | 0.0211 |
| 10 x 10 stress | BFS | 9 | 37 | 0.0631 |

The table is a single illustrative run, not a performance guarantee. Both methods returned equal path costs on the same maps. Expanded-node counts better show the heuristic's effect on these examples than sub-millisecond timings.

## 6. Interface and deployment

The Streamlit app includes three presets, map-size selection, baseline wall editing, live probabilities and risk settings, optional sequential heat evidence, route playback, per-algorithm traces, a same-map A*/BFS table, and an MDP utility chart. `requirements.txt` lists runtime packages. The Dockerfile runs the local app on port 8501, and GitHub Actions installs test dependencies and runs pytest on pushes and pull requests. Deployment is documented but no cloud account or hosted service is provisioned.

## 7. Limitations and future work

The MDP is a one-step expected-reward choice, not a full transition model over time. The simulation assumes binary corridor state, fixed rewards, known sensor rates, conditional independence for the optional second sensor, static walls, and one agent. Future classroom extensions could add a small transition table, multiple monitored corridors, sensor dependence, dynamic fire spread, multiple agents, or multi-floor graphs. Any real evacuation application would require validated building data and qualified safety engineering well beyond this project.

## 8. Defense takeaway

The contribution is the explicit loop between the three algorithms: search proposes a route, Bayes changes belief from evidence, the MDP selects an action at a derived boundary, and search recomputes a safe alternate route after a detour.
