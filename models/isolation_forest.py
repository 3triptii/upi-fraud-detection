import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "amount",
    "transaction_hour",
    "failed_attempts",
    "account_age_days"
]


def main():
    print("Loading UPI transaction data...")

    df = pd.read_csv("data/upi_transactions.csv")

    X = df[FEATURES].copy()

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Training Isolation Forest...")

    model = IsolationForest(
        n_estimators=200,
        contamination=0.10,
        random_state=42
    )

    predictions = model.fit_predict(X_scaled)

    # Isolation Forest: -1 = anomaly, 1 = normal
    df["isolation_forest_anomaly"] = (
        predictions == -1
    ).astype(int)

    print("\n" + "=" * 50)
    print("ISOLATION FOREST RESULTS")
    print("=" * 50)

    anomalies = df["isolation_forest_anomaly"].sum()

    print(f"Total transactions       : {len(df)}")
    print(f"Anomalies detected       : {anomalies}")
    print(f"Actual fraud transactions: {df['is_fraud'].sum()}")

    fraud_captured = df.loc[
        df["isolation_forest_anomaly"] == 1,
        "is_fraud"
    ].sum()

    print(f"Fraud captured            : {fraud_captured}")

    df.to_csv(
        "data/upi_transactions_with_isolation_forest.csv",
        index=False
    )

    print(
        "\nSaved to: "
        "data/upi_transactions_with_isolation_forest.csv"
    )


if __name__ == "__main__":
    main()