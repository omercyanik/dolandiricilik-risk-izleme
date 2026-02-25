import pandas as pd
import numpy as np
import logging
from src.config import DATA_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("starting feature engineering...")

    # time order & typecast
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by="timestamp").reset_index(drop=True)

    # --- time features
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    # 00:00-05:59 and 23:00-23:59 -> unusual
    df["unusual_hour"] = df["hour"].apply(lambda x: 1 if (x < 6 or x >= 23) else 0)

    # --- velocity (customer-level)
    logging.info("calculating velocity features...")

    # sort for rolling window
    df = df.sort_values(by=["customer_id", "timestamp"]).reset_index(drop=True)

    # Use timestamp index for time-based rolling
    df_indexed = df.set_index("timestamp")
    grouped = df_indexed.groupby("customer_id", sort=False)

    # Rolling counts and sums (time-based)
    df["tx_count_1h"] = grouped["amount"].rolling("1h").count().values
    df["tx_count_24h"] = grouped["amount"].rolling("24h").count().values
    df["tx_count_7d"] = grouped["amount"].rolling("7d").count().values

    df["amount_sum_1h"] = grouped["amount"].rolling("1h").sum().values
    df["amount_sum_24h"] = grouped["amount"].rolling("24h").sum().values
    df["amount_sum_7d"] = grouped["amount"].rolling("7d").sum().values

    # Rolling mean/std for z-score (7d)
    df["amount_mean_7d"] = grouped["amount"].rolling("7d").mean().values
    df["amount_std_7d"] = grouped["amount"].rolling("7d").std().values

    # Back to global time order
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Z-score
    df["amount_std_7d"] = df["amount_std_7d"].fillna(1.0).replace(0, 1.0)
    df["amount_zscore_7d"] = (df["amount"] - df["amount_mean_7d"]) / df["amount_std_7d"]
    df["amount_zscore_7d"] = df["amount_zscore_7d"].fillna(0.0)

    # --- advanced behavioral
    logging.info("calculating behavioral factors...")
    
    # ratio: current amount vs 7d running mean
    df["amount_vs_mean_7d_ratio"] = df["amount"] / (df["amount_mean_7d"].replace(0, 1.0) + 1e-5)
    df["amount_vs_mean_7d_ratio"] = df["amount_vs_mean_7d_ratio"].fillna(1.0)

    # burst detection: 1h tx fraction over 24h
    df["tx_velocity_1h_to_24h_ratio"] = df["tx_count_1h"] / (df["tx_count_24h"].replace(0, 1.0) + 1e-5)
    df["tx_velocity_1h_to_24h_ratio"] = df["tx_velocity_1h_to_24h_ratio"].fillna(0.0)


    # --- categorical flags
    logging.info("calculating deviation flags...")

    # First seen country per customer
    first_country = (
        df.groupby(["customer_id", "country"])["timestamp"]
        .min()
        .reset_index()
        .rename(columns={"timestamp": "first_seen_country"})
    )
    df = df.merge(first_country, on=["customer_id", "country"], how="left")
    # ✅ Correct logic: new_country if this tx is the first time that customer uses that country
    df["new_country"] = (df["timestamp"] == df["first_seen_country"]).astype(int)

    # First seen device per customer
    first_device = (
        df.groupby(["customer_id", "device_id"])["timestamp"]
        .min()
        .reset_index()
        .rename(columns={"timestamp": "first_seen_device"})
    )
    df = df.merge(first_device, on=["customer_id", "device_id"], how="left")
    df["new_device"] = (df["timestamp"] == df["first_seen_device"]).astype(int)

    # Optional alias for clarity in UI/features
    df["cross_border"] = df["new_country"]

    # Drop temp columns (keep amount_mean/std? we only need zscore)
    df.drop(
        ["first_seen_country", "first_seen_device", "amount_mean_7d", "amount_std_7d"],
        axis=1,
        inplace=True,
        errors="ignore",
    )

    # --- graph/network features
    logging.info("extracting graph features...")
    
    # device sharing count (expanding sum w/ shift to prevent leakage)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["shared_device_customer_count"] = df.groupby("device_id")["new_device"].transform(
        lambda x: x.shift(1).expanding().sum()
    ).fillna(0)

    # --- historical risk
    logging.info("calculating target-decoded risk ratios...")

    global_fraud_mean = float(df["is_fraud"].mean()) if "is_fraud" in df.columns else 0.0

    # expanding mean, shift(1) to avoid leaking
    df["merchant_risk_7d"] = df.groupby("merchant_id")["is_fraud"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df["merchant_risk_7d"] = df["merchant_risk_7d"].fillna(global_fraud_mean)

    df["device_risk_7d"] = df.groupby("device_id")["is_fraud"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    df["device_risk_7d"] = df["device_risk_7d"].fillna(global_fraud_mean)

    # Final sort (nice for downstream)
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df

def process_and_save():
    input_path = DATA_DIR / "transactions.csv"
    output_path = DATA_DIR / "features.csv"

    df = pd.read_csv(input_path, parse_dates=["timestamp"])
    df_feats = engineer_features(df)

    df_feats["timestamp"] = pd.to_datetime(df_feats["timestamp"]).dt.strftime('%d-%m-%Y %H:%M:%S')
    df_feats.to_csv(output_path, index=False)
    logging.info(f"done. saved to {output_path}")

if __name__ == "__main__":
    process_and_save()