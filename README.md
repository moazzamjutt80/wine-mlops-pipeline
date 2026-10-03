# Wine MLOps Pipeline
![CI](https://github.com/moazzamjutt80/wine-mlops-pipeline/actions/workflows/ci.yml/badge.svg)

## Project Overview
This project is an end-to-end MLOps pipeline for classifying wine cultivars. It demonstrates core MLOps practices, including automated testing, quality gating, model tracking, and CI/CD pipelines, to reliably train, evaluate, and register classification models based on wine dataset features.

## Repository Structure
```
.
├── .github/workflows/   # CI/CD pipelines
├── src/                 # Application source code (data handling, training, evaluation, config)
├── tests/               # Pytest suite including CI quality gates
├── data/                # Data storage directory
├── Makefile             # Convenient commands for automation
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## Setup

### Windows (PowerShell)
```powershell
# Create and activate a Python 3.10 virtual environment
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1

# Run the pipeline steps
make install
make lint
make test
make train
make evaluate
```

### Linux
```bash
# Create and activate a Python 3.10 virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Run the pipeline steps
make install
make lint
make test
make train
make evaluate
```

## MLflow Tracking
To view tracked runs, experiments, and registered models, start the MLflow UI with the following command:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Once it starts, navigate to `http://127.0.0.1:5000` in your web browser.

### Experiment and Model Details
- **Experiment:** `Wine-Cultivar-Classification` — this experiment tracks and groups all of our hyperparameter configurations and cross-validation runs.
- **Model Registration:** The best performing model from the training runs is registered under the name `WineClassifier`. The top model overall is assigned the `champion` alias, making it easy to fetch for evaluation and inference.

## Quality Gates
Our CI pipeline includes strict quality gates to prevent degrading or buggy models from being promoted. The test suite (`make test`) enforces the following rules:
- **Predictive Performance:** The validation macro F1 score must be `>= 0.88`. This ensures no regressions in the model's accuracy and predictive capabilities.
- **Inference Latency:** The median inference batch latency must be `<= 30 ms`. This guarantees the model remains performant enough for real-time predictions.
- **Output Schema Validation:** The model's predictions must be of integer data type and strictly restricted to the class indices `{0, 1, 2}`. This protects downstream applications from malformed outputs.