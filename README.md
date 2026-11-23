# Mini Emergency Exit Planner

A small educational Python application that connects three AI fundamentals in a closed loop:

1. **A*** finds a shortest path through a four-directional grid.
2. **Bayesian inference** updates the probability that a monitored corridor is blocked from one or two hypothetical sensor readings.
3. **A one-step MDP policy** compares proceeding with detouring; a detour marks the monitored corridor blocked and runs A* again to Exit B.

> **Safety:** this is a classroom simulation with invented map, sensor, and reward assumptions. It is not connected to real sensors or a verified building and must never guide a real evacuation.

The supplied course handout allows a console-only implementation and calls deployment optional. This repository provides the planned Streamlit interface, automated tests, report, slide deck, and local/container deployment setup as an expanded deliverable.

## Quick start

Python 3.10+ is recommended.

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit. The app works offline after its Python dependencies have been installed.

Console demonstration:

```powershell
python main.py --sensor alert
python main.py --sensor clear
```

## One-command checks

```powershell
python -m pytest -q
```

The tests cover the expected 5.56% clear and 65.38% alert posteriors, Exit A/Exit B policy outcomes, a noisy sensor crossing the MDP boundary, sequential heat evidence, prior extremes, exact-boundary tie behavior, unreachable paths, and equal A*/BFS optimal path lengths. Benchmark timings vary by host; expanded-node counts and path lengths are the stable comparison.

## Assumptions and equations

The baseline is a 6×6 grid with start `(2, 0)`, Exit A `(2, 5)`, Exit B `(5, 5)`, three walls, and monitored corridor `(2, 3)`. Movement is four-directional at cost 1. The stress grid is 10×10. Coordinates are `(row, column)`.

Default sensor assumptions: prior blockage 0.25; smoke true-positive rate 0.85; smoke false-positive rate 0.15. Optional heat readings are treated as conditionally independent given corridor state, with rates 0.80 and 0.20.

Default rewards: proceed if clear +100; proceed if blocked −100; detour +40. Thus `EU(Proceed) = 100 − 200p`, `EU(Detour) = 40`, and the equality threshold is `p* = 0.30`. At a tie the implementation chooses DETOUR for safety. The UI risk slider changes the blocked-state reward and recalculates the threshold.

## Interactive features

- Clear, Smoke Alert, and Noisy Sensor presets.
- 6×6 baseline and 10×10 stress maps.
- Optional wall editing on the baseline map.
- Live prior, sensor-rate, observation, and risk sliders.
- Optional sequential heat sensor.
- Search trace, Bayesian calculation, MDP utilities, A*/BFS benchmark, and policy-boundary plot.
- Tile-by-tile playback of the selected route.

## Deployment

### Docker

```powershell
docker build -t mini-emergency-exit-planner .
docker run --rm -p 8501:8501 mini-emergency-exit-planner
```

Then open `http://localhost:8501`. For a hosted class demo, push this repository to a Git provider and configure a Streamlit-compatible host to install `requirements.txt` and run `streamlit run app.py`. No secrets or external services are required. Hosting is not automatically provisioned by this repository.

## Repository map

- `app.py`: Streamlit app and visualization.
- `pipeline.py`: A* → Bayes → MDP → optional A* reroute.
- `algorithms/`: independent algorithm modules.
- `config.py`: labeled hypothetical inputs.
- `main.py`: offline console demo and benchmark.
- `tests/`: pytest coverage.
- `docs/`: technical report, presentation, benchmark results, AI-use note, contribution template, and defense/demo script.

## Defense/demo

Use `docs/demo_and_defense_script.md` for the 2–3 minute demo and likely Q&A. Replace the author placeholder in the presentation and contribution template. Review and understand all algorithm code before presenting; the AI-use note describes assistance accurately.
