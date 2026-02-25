import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytest
from src.feature_engineering import engineer_features

def test_engineer_features_temporal_leakage():
    # Create simple data simulating transactions
    # To test if expanding window (merchant_risk) leaks current time point
    data = pd.DataFrame({
        "timestamp": [
            datetime(2023, 1, 1, 10, 0),
            datetime(2023, 1, 2, 10, 0),
            datetime(2023, 1, 3, 10, 0),
        ],
        "customer_id": ["C1", "C1", "C1"],
        "merchant_id": ["M1", "M1", "M1"],
        "device_id": ["D1", "D1", "D1"],
        "amount": [100.0, 200.0, 300.0],
        "country": ["TR", "TR", "TR"],
        "is_fraud": [1, 0, 1]
    })
    
    # Run feature engineering
    df_feats = engineer_features(data)
    
    # 1. Check if first row has global/default merchant risk (since no past data)
    global_mean = data["is_fraud"].mean()
    assert df_feats.loc[0, "merchant_risk_7d"] == global_mean, "First row should have global mean, not target leakage."
    
    # 2. Check if second row only sees the first row
    assert df_feats.loc[1, "merchant_risk_7d"] == 1.0, "Second row should only see target=1 from first row."
    
    # 3. Check if third row sees the mean of first two rows
    assert df_feats.loc[2, "merchant_risk_7d"] == 0.5, "Third row should see mean of first two targets (1, 0)."

def test_engineer_features_velocity():
    data = pd.DataFrame({
        "timestamp": [
            datetime(2023, 1, 1, 10, 0),
            datetime(2023, 1, 1, 10, 30), # 30 min later
            datetime(2023, 1, 1, 12, 30), # 2.5 hours later
        ],
        "customer_id": ["C1", "C1", "C1"],
        "merchant_id": ["M1", "M1", "M2"],
        "device_id": ["D1", "D1", "D2"],
        "amount": [10.0, 20.0, 30.0],
        "country": ["TR", "TR", "US"],
        "is_fraud": [0, 0, 0]
    })
    
    df_feats = engineer_features(data)
    
    # 1st row: 1 hour window count should be 1
    assert df_feats.loc[0, "tx_count_1h"] == 1
    # 2nd row: 1 hour window count should be 2
    assert df_feats.loc[1, "tx_count_1h"] == 2
    # 3rd row: 1 hour window count should be 1 (2.5h later)
    assert df_feats.loc[2, "tx_count_1h"] == 1
    
    # Test new country
    assert df_feats.loc[0, "new_country"] == 1 # TR first time
    assert df_feats.loc[1, "new_country"] == 0 # TR second time
    assert df_feats.loc[2, "new_country"] == 1 # US first time
