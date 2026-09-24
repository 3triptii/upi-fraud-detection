import pandas as pd


def detect_iqr_anomalies(df, column="amount"):
    """
    Detect unusual transaction amounts using the IQR method.
    """

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df["iqr_anomaly"] = (
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ).astype(int)

    return df, lower_bound, upper_bound


def main():
    print("Loading UPI transaction data...")

    df = pd.read_csv("data/upi_transactions.csv")

    print(f"Total transactions: {len(df)}")

    df, lower, upper = detect_iqr_anomalies(df)

    print("\n" + "=" * 50)
    print("IQR ANOMALY DETECTION")
    print("=" * 50)

    print(f"Lower bound : ₹{lower:.2f}")
    print(f"Upper bound : ₹{upper:.2f}")

    anomalies = df["iqr_anomaly"].sum()

    print(f"IQR anomalies detected: {anomalies}")

    actual_fraud = df["is_fraud"].sum()
    fraud_in_anomalies = df.loc[
        df["iqr_anomaly"] == 1,
        "is_fraud"
    ].sum()

    print(f"Actual fraud transactions: {actual_fraud}")
    print(f"Fraud captured by IQR: {fraud_in_anomalies}")

    df.to_csv("data/upi_transactions_with_iqr.csv", index=False)

    print("\nSaved to: data/upi_transactions_with_iqr.csv")


if __name__ == "__main__":
    main()