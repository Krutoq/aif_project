# Live demo and defense notes

## 2–3 minute demonstration

1. **0:00–0:20:** Introduce the toy problem and state that all map, sensor, and reward values are assumed for a classroom simulation.
2. **0:20–0:55:** Select **Clear**. Show Bayes reducing blockage belief to about 5.6%, the MDP choosing PROCEED, and the route to Exit A. Expand the A* and Bayes traces.
3. **0:55–1:35:** Select **Smoke alert**. Show the posterior near 65.4%, expected utility comparison, DETOUR decision, and A* path to Exit B around the sensor corridor. Play the evacuation animation.
4. **1:35–2:10:** Select **Noisy sensor** or adjust false-positive rate and show how posterior and policy respond. Point to the boundary chart and A*/BFS table.
5. **2:10–2:35:** State the model limits: fixed grid, hypothetical independent sensor model, one-step reward comparison, no real sensors or building validation.

## Short Q&A

**Why A*?** Its Manhattan heuristic guides search to the goal; with four-way unit-cost moves it does not overestimate, preserving a shortest route.

**Why can a noisy alert still lead to PROCEED?** Bayes weighs the alert's likelihood against its false-positive rate and the prior. The MDP then compares the resulting expected utilities.

**Why is the baseline boundary 30%?** Set `100 − 200p = 40`; solving gives `p = 0.30`. Increasing the loss for proceeding into a blocked corridor moves this threshold lower.

**What does the heat sensor assume?** Conditional independence from the smoke sensor given the true corridor state. Real sensor dependencies would need measured data.

**Can this be used in an actual building?** No. It lacks validated floorplans, live sensing, fire dynamics, human factors, code review, and life-safety certification.
