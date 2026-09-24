import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


DATA_PATH = "data/upi_transactions.csv"

FEATURES = [
    "amount",
    "transaction_hour",
    "merchant_category",
    "device_type",
    "location",
    "failed_attempts",
    "account_age_days"
]

TARGET = "is_fraud"


def main():

    print("Loading UPI transaction dataset...")

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET]

    categorical_features = [
        "merchant_category",
        "device_type",
        "location"
    ]

    numerical_features = [
        "amount",
        "transaction_hour",
        "failed_attempts",
        "account_age_days"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            ),
            (
                "numerical",
                "passthrough",
                numerical_features
            )
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    print("\nTraining Random Forest...")

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 50)
    print("FRAUD DETECTION MODEL RESULTS")
    print("=" * 50)

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    auc = roc_auc_score(y_test, probabilities)

    print(f"\nROC-AUC Score: {auc:.4f}")

    import joblib

    joblib.dump(
        pipeline,
        "models/upi_fraud_model.pkl"
    )

    print("\nModel saved to:")
    print("models/upi_fraud_model.pkl")


if __name__ == "__main__":
    main()