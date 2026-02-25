import pandas as pd
import numpy as np
import datetime
import uuid
import logging
from tqdm import tqdm
from scipy.stats import lognorm

from src.config import DATA_DIR, NUM_CUSTOMERS, NUM_MERCHANTS, NUM_DEVICES, DAYS_OF_DATA, FRAUD_RATE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_profiles(n_customers, n_merchants, n_devices):
    np.random.seed(42)
    # Customers
    customers = pd.DataFrame({
        "customer_id": [f"CUST_{i}" for i in range(n_customers)],
        "mean_amount": np.random.uniform(20, 150, n_customers),
        "std_amount": np.random.uniform(5, 50, n_customers),
        "home_country": np.random.choice(["TR", "US", "GB", "DE", "FR"], n_customers, p=[0.7, 0.1, 0.1, 0.05, 0.05])
    })
    
    # Merchants
    merchants = pd.DataFrame({
        "merchant_id": [f"MERCH_{i}" for i in range(n_merchants)],
        "merchant_country": np.random.choice(["TR", "US", "GB", "DE", "FR", "RU", "CN"], n_merchants, p=[0.6, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05]),
        "fraud_risk_factor": np.random.beta(1, 10, n_merchants) 
    })
    
    # Devices
    devices = [f"DEV_{i}" for i in range(n_devices)]
    
    return customers, merchants, devices

def generate_transactions(customers, merchants, devices, days=DAYS_OF_DATA, base_fraction=1.0):
    start_date = datetime.datetime.now() - datetime.timedelta(days=days)
    
    n_customers = len(customers)
    n_merchants = len(merchants)
    
    channels = ["POS", "ECOM", "TRANSFER"]
    
    records = []
    
    # Approximating a total of ~500k-1M transactions
    total_tx = int(1000000 * base_fraction)
    
    logging.info("Generating regular transactions...")
    
    # Generate random timestamps over the duration
    random_seconds = np.random.randint(0, days * 24 * 3600, total_tx)
    random_seconds.sort() # Time-ordered
    
    timestamps = [start_date + datetime.timedelta(seconds=int(s)) for s in random_seconds]
    
    # Random assignments
    cust_indices = np.random.randint(0, n_customers, total_tx)
    merch_indices = np.random.randint(0, n_merchants, total_tx)
    
    cust_df = customers.iloc[cust_indices].reset_index(drop=True)
    merch_df = merchants.iloc[merch_indices].reset_index(drop=True)
    
    # Normal amounts using lognormal centered around customer's mean
    sigma = 0.5
    scale = cust_df["mean_amount"].values
    amounts = np.random.lognormal(mean=np.log(scale), sigma=sigma)
    amounts = np.round(np.clip(amounts, 1.0, 10000.0), 2)
    
    # Channel distribution
    channel_choices = np.random.choice(channels, total_tx, p=[0.5, 0.4, 0.1])
    
    # Default: device mapping 1:1 with customer mostly
    device_choices = np.array([f"DEV_{c}" for c in cust_indices])
    
    # Introduce label
    labels = np.zeros(total_tx, dtype=int)
    
    df = pd.DataFrame({
        "tx_id": [str(uuid.uuid4()) for _ in range(total_tx)],
        "timestamp": timestamps,
        "customer_id": cust_df["customer_id"],
        "merchant_id": merch_df["merchant_id"],
        "device_id": device_choices,
        "channel": channel_choices,
        "amount": amounts,
        "country": cust_df["home_country"], # Generally local
        "is_fraud": labels
    })
    
    logging.info("Injecting fraud scenarios...")
    
    num_frauds = int(total_tx * FRAUD_RATE)
    fraud_indices = np.random.choice(df.index, num_frauds, replace=False)
    
    for idx in fraud_indices:
        df.at[idx, "is_fraud"] = 1
        
        # Fraud scenario types
        scenario = np.random.choice(["burst", "cross_border", "hijack", "high_amount"])
        
        # Burst: duplicate transaction multiple times within minutes
        if scenario == "burst":
            pass # difficult in simple list map, we will just simulate anomalies
            
        if scenario == "cross_border":
            df.at[idx, "country"] = np.random.choice(["RU", "CN", "NG"])
            df.at[idx, "channel"] = "ECOM"
            df.at[idx, "amount"] = df.at[idx, "amount"] * np.random.uniform(2, 5)
            
        elif scenario == "hijack":
            df.at[idx, "device_id"] = np.random.choice(devices) # new device
            df.at[idx, "channel"] = "TRANSFER"
            df.at[idx, "amount"] = df.at[idx, "amount"] * np.random.uniform(5, 10)
            
        elif scenario == "high_amount":
            df.at[idx, "amount"] = df.at[idx, "amount"] * np.random.uniform(10, 50)
            
    # Add label noise
    noise_idx = np.random.choice(df.index, int(total_tx * 0.0005), replace=False)
    df.loc[noise_idx, "is_fraud"] = 1 - df.loc[noise_idx, "is_fraud"]
    
    # Sort to be absolutely sure
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    return df

def generate_and_save():
    logging.info("Starting synthetic data generation...")
    customers, merchants, devices = generate_profiles(NUM_CUSTOMERS, NUM_MERCHANTS, NUM_DEVICES)
    
    # using base_fraction=0.2 to generated 200k rows (faster for dev)
    df = generate_transactions(customers, merchants, devices, base_fraction=0.3)
    
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%d-%m-%Y %H:%M:%S')
    
    path = DATA_DIR / "transactions.csv"
    df.to_csv(path, index=False)
    
    logging.info(f"Generated {len(df)} transactions. Saved to {path}")
    logging.info(f"Fraud rate: {df['is_fraud'].mean():.4f}")

if __name__ == "__main__":
    generate_and_save()
