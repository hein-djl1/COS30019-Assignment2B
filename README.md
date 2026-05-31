# COS30019 Assignment 2B - TBRGS

## Run the app

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python run_tbrgs.py
```

Open this in your browser:

```text
http://127.0.0.1:8000
```

Stop the server with `Ctrl+C`.

## Optional: if required files are missing

If processed data is missing:

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

Prediction done in:

```text
src/services/prediction.py
```

## Configuration

Defaults are stored in:

```text
config/tbrgs_config.json
```

## Known Limitations

- The current traffic estimate is static for the selected departure timestamp, not time-dependent as the vehicle traverses each edge.
- The SCATS graph is inferred from SCATS site labels and coordinates because no explicit road-edge shapefile is included.
- Some graph filtering is manual to keep the displayed network focused and usable.

