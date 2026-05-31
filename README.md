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

If model files are missing:

```powershell
.\.venv\Scripts\python src\models\xgboost_model.py
.\.venv\Scripts\python src\models\gru_model.py
.\.venv\Scripts\python src\models\lstm_model.py
```
