# AI Emergency Medical Dispatch Decision Support System

A desktop prototype for emergency medical dispatch. The full application uses
the Cambridge, UK road network and can load a local KTAS triage dataset. When
that dataset is absent, it falls back to synthetic examples. The Final Project
demo below is a deterministic, self-contained A* + Bayes + MDP scenario.

## Final Project course demonstration

The Course Demo page in the desktop GUI and the terminal command both run the
same small, repeatable ambulance-route scenario. It connects one algorithm
from each required topic group: A* search, Bayesian reasoning about a noisy
road-closure alert, and a one-step Markov Decision Process (MDP) that chooses
whether to continue or reroute. When rerouting is selected, A* excludes the
risk edge and finds the best available alternative.

Run both examples from this directory:

```bash
python final_project_demo.py
```

Or run only one sensor scenario:

```bash
python final_project_demo.py --signal alert
python final_project_demo.py --signal clear
```

On Windows, `run_final_project_demo.bat` starts both scenarios in a console.

The demonstration uses only the Python standard library and does not download
the Cambridge map, load patient data, or require the desktop GUI. It shows the
input, important intermediate calculations, algorithm outputs, and a short
interpretation. All sensor probabilities, transition probabilities, rewards,
and toy road lengths are explicitly illustrative; they are not live traffic
measurements or a real emergency-routing safety policy.

### Algorithms and assumptions

1. **A* search** — finds a shortest path by road length. Its straight-line
   distance heuristic is admissible on the toy graph. `ai/routing.py` also
   supports optional directed blocked-edge pairs so the demo can recompute a
   route without mutating the original graph.
2. **Bayesian inference** — uses prior closure probability 0.20, sensor
   sensitivity 0.80, and false-positive rate 0.10. With an alert, the posterior
   closure probability is 66.7%; without an alert it is 5.3%.
3. **One-step MDP** — uses states Clear/Blocked, actions Continue/Reroute, a
   stated transition table, illustrative rewards, discount 0.90, and next-state
   values. It selects the action with the highest belief-weighted Bellman value.

### Example result

```text
NO ALERT: P(blocked | observation) = 5.3%
  Continue value = 6.98; reroute value = -11.85
  Decision: CONTINUE; route AMB -> JCT -> CALL (200 m)

ALERT: P(blocked | observation) = 66.7%
  Continue value = -49.27; reroute value = -11.85
  Decision: REROUTE; route AMB -> A -> B -> CALL (420 m)
```

For a short live demo, open **Course Demo**, run **No alert**, and explain why
the low posterior risk keeps the shortest route. Switch to **Road alert** and
show how the updated belief changes the MDP choice and causes A* to avoid the
risk edge. The CLI remains available for a deterministic text trace.

The original full desktop decision-support application remains available via
`python app.py`; install `requirements.txt` and run `python setup_data.py` first
for its Cambridge map workflow.

If the directed Cambridge graph has no path for an ambulance-call pair, the
assignment stage uses a penalized straight-line estimate to keep the matching
finite. The map does not draw a route for that pair and marks the affected call.
Cached triage models are retrained automatically when their scikit-learn
version differs from the installed version.

The editable English presentation, including speaker notes and a two-minute
live-demo outline, is in `presentation/Ambulance_Route_AI_Final_Project_v2.pptx`.

## What it does

1. **Triage** — Predicts patient priority (Critical / High / Medium / Low) from
   age, vital signs and observations using a Random Forest classifier. It uses
   the KTAS dataset when supplied locally and synthetic data otherwise.
2. **Ambulance assignment** — Assigns available ambulances to calls to minimise
   total response time via the Hungarian algorithm.
3. **Routing** — Computes optimal ambulance routes on the real Cambridge road
   network using A* (vs Dijkstra baseline).
4. **Hospital recommendation** — Recommends the best hospital per patient using
   a hard specialty-match constraint followed by distance/capacity scoring.
5. **Analytics** — Confusion matrix, accuracy/F1/recall, response-time charts,
   and algorithm comparisons all displayed on the Analytics page.

## Setup (run once, requires internet)

```bash
pip install -r requirements.txt
python setup_data.py
```

This downloads and caches the Cambridge road network graph locally.

**Note:** The Interactive Map page always requires an internet connection to load
Leaflet JS/CSS and OpenStreetMap tiles.  Only the road-network graph used by the
AI algorithms is cached locally.

## Running the app

```bash
python app.py
```

## Data

`data_files/ktas_data.csv` is an optional local input for triage training. It is
not included in this repository; without it, the application uses synthetic
data. The fitted `triage_model.pkl`, downloaded Cambridge graph, and runtime
caches are generated locally and are also excluded. Do not commit patient-level
clinical data or models trained on it to a public repository.

## Project structure

```
app.py                      Entry point
setup_data.py               One-time network download
data_files/
    ktas_data.csv           Real KTAS triage dataset
    cambridge_network.graphml  Cached after setup_data.py runs
data/
    medical_data.py         Real-data loader + synthetic fallback
    scenario_generator.py   Emergency-call and ambulance placement
    hospitals.py            Cambridge hospital definitions
    ambulance_stations.py   Ambulance station definitions
    network_loader.py       OSMnx graph loader / cache
ai/
    triage.py               Random Forest triage classifier
    assignment.py           Hungarian algorithm assignment
    routing.py              A* and Dijkstra routing
    road_risk.py             Bayesian road-closure inference
    route_mdp.py             One-step MDP route decision
    hospital.py             Multi-criteria hospital recommendation
    pipeline.py             Full pipeline orchestration
final_project_demo.py       Course demo connecting Bayes, MDP, and A*
gui/
    main_window.py          Main window + page navigation
    dashboard.py            Scenario controls and status
    map_widget.py           Leaflet interactive map
    analytics.py            Analytics charts and metrics
utils/
    metrics.py              Timing utilities
```

## Requirements

- Python 3.12+
- See `requirements.txt` for library versions
