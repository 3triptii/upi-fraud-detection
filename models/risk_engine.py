import pandas as pd


def calculate_risk(row):
    score = 0

    # IQR amount anomaly
    if row["iqr_anomaly"] == 1:
        score += 30

    # Isolation Forest anomaly
    if row["isolation_forest_anomaly"] == 1:
        score += 40

    # Unusual transaction time
    if row["transaction_hour"] in [0, 1, 2, 3, 4]:
        score += 15

    # Multiple failed attempts
    if row["failed_attempts"] >= 3:
        score += 15

    # New account
    if row["account_age_days"] < 90:
        score += 10

    return min(score, 100)


def get_risk_level(score):
    if score >= 70:
        return "HIGH"
    elif score >= 40:
        return "MEDIUM"
    else:
        return "LOW"


def main():
    print("Loading anomaly detection results...")

    df = pd.read_csv(
        "data/upi_transactions_with_isolation_forest.csv"
    )

    # Recalculate IQR flag
    Q1 = df["amount"].quantile(0.25)
    Q3 = df["amount"].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df["iqr_anomaly"] = (
        (df["amount"] < lower_bound)
        | (df["amount"] > upper_bound)
    ).astype(int)

    # Calculate risk score
    df["risk_score"] = df.apply(calculate_risk, axis=1)

    df["risk_level"] = df["risk_score"].apply(get_risk_level)

    # Final fraud flag
    df["fraud_flag"] = (
        df["risk_score"] >= 70
    ).astype(int)

    print("\n" + "=" * 50)
    print("UPI FRAUD RISK ENGINE")
    print("=" * 50)

    print(f"Total transactions : {len(df)}")
    print(f"High-risk          : {(df['risk_level'] == 'HIGH').sum()}")
    print(f"Medium-risk        : {(df['risk_level'] == 'MEDIUM').sum()}")
    print(f"Low-risk           : {(df['risk_level'] == 'LOW').sum()}")

    print(
        f"\nFraud transactions detected: "
        f"{df['fraud_flag'].sum()}"
    )

    df.to_csv(
        "data/upi_risk_scored.csv",
        index=False
    )

    print("\nSaved to: data/upi_risk_scored.csv")


if __name__ == "__main__":
    main()