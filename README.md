# COS30019 Assignment 2B - TBRGS

Traffic-Based Route Guidance System for the Boroondara SCATS dataset.

The current implementation integrates:

- SCATS data preprocessing
- An XGBoost traffic-flow prediction model using `scats_site + location`
- LSTM and GRU traffic-flow prediction models
- Travel-time conversion from predicted traffic flow
- Part A A* search from the `Search/` folder
- A browser GUI with model selection, route table, and highlighted graph path

## Quick Start

From the repository root:

```powershell
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python run_tbrgs.py
```

Then open:

```text
http://127.0.0.1:8000
```

The GUI lets you choose:

- ML model
- origin SCATS site and location
- destination SCATS site and location
- date and time
- number of routes to return

The graph highlights the shortest A* path and shows the estimated edge cost on each highlighted edge.

## Project Structure

```text
Search/                  Part A search algorithms and graph classes
Info/                    Assignment data files, ignored by git
data/processed/          Cleaned SCATS data and generated train/validation/test splits
data/processed_location/ Location-level train/validation/test splits for XGBoost
data/models/             Trained model artifacts
src/data/                Preprocessing code
src/models/              Model training code
src/services/            Model registry and prediction interface
src/integration/         SCATS graph, edge-cost, and routing integration
src/web_tbrgs.py         Browser GUI and API server
run_tbrgs.py             GUI runner
config/tbrgs_config.json GUI defaults
```

## Data Processing

To regenerate processed data:

```powershell
.\.venv\Scripts\python src\data\preprocess_scats.py
```

This reads:

```text
Info/Scats Data October 2006.xls
```

and writes:

- `data/processed/cleaned_scats_long.csv`
- `data/processed/train.csv`
- `data/processed/validation.csv`
- `data/processed/test.csv`
- `data/processed/scats_sequences.npz`
- `data/processed/metadata.json`

The processed data includes SCATS site location, latitude, and longitude, so the models and graph can use location information.

To keep each SCATS location separate instead of collapsing all locations under a site:

```powershell
.\.venv\Scripts\python src\data\preprocess_scats.py --series-key location --output-dir data\processed_location
```

## Train the XGBoost Model

```powershell
.\.venv\Scripts\python src\models\xgboost_model.py
```

Outputs are saved to:

```text
data/models/xgboost/
```

Current model features include:

- recent flow values
- hourly/simple rolling flow
- time features
- SCATS site one-hot features
- latitude and longitude
- location one-hot features

## Routing Logic

The route calculation is:

1. Load filtered SCATS sites from processed data.
2. Build a road-like SCATS graph from location names and coordinates.
3. Predict traffic flow for each edge's starting SCATS site.
4. Convert predicted flow to speed.
5. Calculate edge travel time:

```text
edge time = distance / speed + 30 seconds intersection delay
```

6. Build a `Search.Graph`.
7. Run the Part A A* implementation from:

```text
Search/algorithms/astar.py
```

The current implementation uses a static traffic snapshot for the selected departure time.

## Graph Filtering

The GUI graph is intentionally filtered to focus on the working Boroondara network area.

Currently excluded:

- `2825`
- `2827`
- `3812`
- `2846`
- `2200`
- `4321`
- sites west of `4272`, except `4273`

The real connecting site between `4043` and `4272` is kept as `4273`, producing:

```text
4043 -> 4273 -> 4272
```

Filtering lives in:

```text
src/integration/scats_network.py
```

## Adding More Models

Future LSTM/GRU models should plug into the registry in:

```text
src/services/prediction.py
```

Each model adapter should implement:

```python
def predict(self, request: PredictionRequest) -> PredictionResult:
    ...
```

Add the model class to the registry factories so the browser UI model dropdown can use it without changing the routing code.

## Configuration

Defaults are stored in:

```text
config/tbrgs_config.json
```

Example:

```json
{
  "host": "127.0.0.1",
  "port": 8000,
  "default_model": "xgboost",
  "default_date": "2006-10-23",
  "default_time": "08:00",
  "default_top_k": 5
}
```

## Known Limitations

- The current traffic estimate is static for the selected departure timestamp, not time-dependent as the vehicle traverses each edge.
- The SCATS graph is inferred from SCATS site labels and coordinates because no explicit road-edge shapefile is included.
- Some graph filtering is manual to keep the displayed network focused and usable.
- The XGBoost model uses a heuristic to choose the source location for each graph edge because no explicit lane/turn-edge mapping is included.
