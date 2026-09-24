import pandas as pd
import joblib

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

DATA_PATH = "data/upi_transactions.csv"
MODEL_PATH = "models/upi_fraud_model.pkl"


def main():

    print("Loading model and test data...")

    df = pd.read_csv(DATA_PATH)

    model = joblib.load(MODEL_PATH)

    features = [
        "amount",
        "transaction_hour",
        "merchant_category",
        "device_type",
        "location",
        "failed_attempts",
        "account_age_days"
    ]

    X = df[features]
    y = df["is_fraud"]

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    print("\n" + "=" * 50)
    print("UPI FRAUD MODEL EVALUATION")
    print("=" * 50)

    print(f"Precision : {precision_score(y, predictions):.4f}")
    print(f"Recall    : {recall_score(y, predictions):.4f}")
    print(f"F1 Score  : {f1_score(y, predictions):.4f}")
    print(f"ROC-AUC   : {roc_auc_score(y, probabilities):.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y, predictions))


if __name__ == "__main__":
    main()