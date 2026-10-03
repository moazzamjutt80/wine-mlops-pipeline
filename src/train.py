import mlflow
import numpy as np
from mlflow.client import MlflowClient
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold

from src.data import get_splits

RANDOM_FOREST_CONFIGS = [
    {"n_estimators": 75, "max_depth": 3},
    {"n_estimators": 150, "max_depth": 6},
    {"n_estimators": 200, "max_depth": None},
    {"n_estimators": 120, "max_depth": 8},
]

BEST_PARAMS = {"n_estimators": 150, "max_depth": 6}

GRADIENT_BOOSTING_CONFIGS = [
    {"n_estimators": 60, "learning_rate": 0.1, "max_depth": 2},
    {"n_estimators": 120, "learning_rate": 0.05, "max_depth": 2},
    {"n_estimators": 80, "learning_rate": 0.2, "max_depth": 3},
    {"n_estimators": 150, "learning_rate": 0.1, "max_depth": 3},
]


def run_cv(model, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    metrics = {
        "train_f1_macro": [],
        "train_accuracy": [],
        "train_log_loss": [],
        "val_f1_macro": [],
        "val_accuracy": [],
        "val_log_loss": [],
    }

    for train_idx, val_idx in cv.split(X, y):
        X_train_f, X_val_f = X.iloc[train_idx], X.iloc[val_idx]
        y_train_f, y_val_f = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train_f, y_train_f)

        # Train metrics
        y_train_pred = model.predict(X_train_f)
        y_train_prob = model.predict_proba(X_train_f)
        metrics["train_f1_macro"].append(f1_score(y_train_f, y_train_pred, average="macro"))
        metrics["train_accuracy"].append(accuracy_score(y_train_f, y_train_pred))
        metrics["train_log_loss"].append(log_loss(y_train_f, y_train_prob, labels=[0, 1, 2]))

        # Val metrics
        y_val_pred = model.predict(X_val_f)
        y_val_prob = model.predict_proba(X_val_f)
        metrics["val_f1_macro"].append(f1_score(y_val_f, y_val_pred, average="macro"))
        metrics["val_accuracy"].append(accuracy_score(y_val_f, y_val_pred))
        metrics["val_log_loss"].append(log_loss(y_val_f, y_val_prob, labels=[0, 1, 2]))

    return {k: np.mean(v) for k, v in metrics.items()}


def train_and_log(model_family, config, X_train, y_train):
    if model_family == "RandomForest":
        model = RandomForestClassifier(**config, random_state=42)
        run_name = f"rf_n{config['n_estimators']}_d{config['max_depth']}"
    else:
        model = GradientBoostingClassifier(**config, random_state=42)
        run_name = (
            f"gb_n{config['n_estimators']}_lr{config['learning_rate']}_d{config['max_depth']}"
        )

    with mlflow.start_run(run_name=run_name) as run:
        # Cross Validation
        cv_metrics = run_cv(model, X_train, y_train)

        # Log params, metrics, tags
        mlflow.log_param("model_family", model_family)
        mlflow.log_params(config)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("random_state", 42)
        mlflow.log_metrics(cv_metrics)
        mlflow.set_tag("model_family", model_family)
        mlflow.set_tag("config_name", run_name)

        # Refit on full train
        model.fit(X_train, y_train)

        input_example = X_train.head(5)
        signature = infer_signature(input_example, model.predict(input_example))

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=input_example,
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        return {
            "run_id": run.info.run_id,
            "run_name": run_name,
            "model_uri": model_info.model_uri,
            "model_family": model_family,
            "params": str(config),
            **cv_metrics,
        }


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Wine-Cultivar-Classification")

    X_train, _, y_train, _ = get_splits()

    results = []

    for config in RANDOM_FOREST_CONFIGS:
        res = train_and_log("RandomForest", config, X_train, y_train)
        results.append(res)

    for config in GRADIENT_BOOSTING_CONFIGS:
        res = train_and_log("GradientBoosting", config, X_train, y_train)
        results.append(res)

    # Find champion
    best_run = max(results, key=lambda x: x["val_f1_macro"])
    print(
        f"\nRegistering champion model from run {best_run['run_name']} "
        f"with val_f1_macro {best_run['val_f1_macro']:.4f}"
    )

    model_version = mlflow.register_model(
        model_uri=best_run["model_uri"], name="WineClassifier"
    )

    client = MlflowClient()
    client.set_registered_model_alias(
        "WineClassifier", "champion", model_version.version
    )

    # Print markdown table
    print(
        "\n| Model Family | Params | Train F1 | Val F1 | "
        "Train Acc | Val Acc | Train LogLoss | Val LogLoss |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for r in results:
        print(
            f"| {r['model_family']} | {r['params']} | "
            f"{r['train_f1_macro']:.4f} | {r['val_f1_macro']:.4f} | "
            f"{r['train_accuracy']:.4f} | {r['val_accuracy']:.4f} | "
            f"{r['train_log_loss']:.4f} | {r['val_log_loss']:.4f} |"
        )

    print(f"\nChampion Run: {best_run['run_name']} (Model Version: {model_version.version})")


if __name__ == "__main__":
    main()
