import numpy as np
import pandas as pd

np.random.seed(42)

N = 10000

data = pd.DataFrame({
    "transaction_id": [f"TXN{i:06d}" for i in range(1, N + 1)],
    "amount": np.round(np.random.lognormal(mean=5, sigma=1.2, size=N), 2),
    "transaction_hour": np.random.randint(0, 24, N),
    "merchant_category": np.random.choice(
        ["Shopping", "Food", "Travel", "Bills", "Entertainment", "Other"],
        N
    ),
    "device_type": np.random.choice(
        ["Android", "iOS", "Web"],
        N,
        p=[0.55, 0.35, 0.10]
    ),
    "location": np.random.choice(
        ["Delhi", "Mumbai", "Bangalore", "Hyderabad", "Chennai", "Pune"],
        N
    ),
    "failed_attempts": np.random.poisson(0.4, N),
    "account_age_days": np.random.randint(30, 3000, N),
})

# Create fraud labels using unusual transaction behaviour
fraud_score = (
    (data["amount"] > data["amount"].quantile(0.97)).astype(int)
    + (data["transaction_hour"].isin([0, 1, 2, 3, 4])).astype(int)
    + (data["failed_attempts"] >= 3).astype(int)
    + (data["account_age_days"] < 90).astype(int)
)

data["is_fraud"] = (fraud_score >= 2).astype(int)

# Ensure approximately 10% fraudulent transactions
fraud_indices = data.index[data["is_fraud"] == 1]

if len(fraud_indices) > 1000:
    keep = np.random.choice(fraud_indices, 1000, replace=False)
    data["is_fraud"] = 0
    data.loc[keep, "is_fraud"] = 1

elif len(fraud_indices) < 1000:
    normal_indices = data.index[data["is_fraud"] == 0]
    extra = np.random.choice(
        normal_indices,
        1000 - len(fraud_indices),
        replace=False
    )
    data.loc[extra, "is_fraud"] = 1

data.to_csv("data/upi_transactions.csv", index=False)

print("=" * 50)
print("UPI TRANSACTION DATASET CREATED")
print("=" * 50)
print(f"Total transactions : {len(data)}")
print(f"Fraud transactions  : {data['is_fraud'].sum()}")
print(f"Normal transactions : {(data['is_fraud'] == 0).sum()}")
print(f"Fraud rate          : {data['is_fraud'].mean() * 100:.2f}%")
print("\nSaved to: data/upi_transactions.csv")