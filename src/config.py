import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Data generation parameters
NUM_CUSTOMERS = 5000
NUM_MERCHANTS = 1000
NUM_DEVICES = 6000
DAYS_OF_DATA = 90
FRAUD_RATE = 0.005  # 0.5% fraud rate

# Model parameters
TARGET_COL = "is_fraud"
NUMERIC_FEATURES = [
    "amount", 
    "tx_count_1h", "tx_count_24h", "tx_count_7d",
    "amount_sum_1h", "amount_sum_24h", "amount_sum_7d",
    "amount_zscore_7d",
    "merchant_risk_7d", "device_risk_7d",
    "shared_device_customer_count",
    "amount_vs_mean_7d_ratio", "tx_velocity_1h_to_24h_ratio"
]
CATEGORICAL_FEATURES = [
    "channel", "country", "new_device", "new_country", "unusual_hour"
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
