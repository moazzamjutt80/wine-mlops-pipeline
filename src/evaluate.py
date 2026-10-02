import mlflow
from sklearn.metrics import accuracy_score, f1_score, log_loss

from src.data import get_splits


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    _, X_test, _, y_test = get_splits()

    model_uri = "models:/WineClassifier@champion"
    print(f"Loading model from {model_uri}...")
    model = mlflow.sklearn.load_model(model_uri)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    f1 = f1_score(y_test, y_pred, average="macro")
    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_prob, labels=[0, 1, 2])

    print("\nTest Metrics (Champion Model):")
    print(f"Macro F1: {f1:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Log Loss: {ll:.4f}")


if __name__ == "__main__":
    main()
