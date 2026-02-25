import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import timedelta
import logging

def generate_forecast(data_path, days=30):
    """
    Reads transaction data, calculates daily fraud volume, 
    and forecasts future volumes using LightGBM.
    Returns: (historical_daily_fraud, forecasted_daily_fraud)
    """
    try:
        # Sadece gerekli kolonları yükle
        df = pd.read_csv(data_path, usecols=['timestamp', 'is_fraud'])
    except Exception as e:
        logging.error(f"Error loading data for forecast: {e}")
        return pd.DataFrame(), pd.DataFrame()
        
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sadece dolandırıcılıkları filtrele ve günlük resample yap (Fraud count)
    df_fraud = df[df['is_fraud'] == 1].copy()
    if df_fraud.empty:
        return pd.DataFrame(), pd.DataFrame()
        
    df_fraud.set_index('timestamp', inplace=True)
    daily_fraud = df_fraud.resample('D').size().rename('fraud_count').reset_index()
    
    # Seri içindeki eksik günleri (fraud olmayan) 0 ile doldur
    full_range = pd.date_range(start=daily_fraud['timestamp'].min(), 
                               end=daily_fraud['timestamp'].max(), freq='D')
    daily_fraud = daily_fraud.set_index('timestamp').reindex(full_range, fill_value=0).reset_index()
    daily_fraud.rename(columns={'index': 'timestamp'}, inplace=True)
    
    # LightGBM için zaman öznitelikleri üretimi
    def create_features(df_input):
        df_feat = df_input.copy()
        df_feat['dayofweek'] = df_feat['timestamp'].dt.dayofweek
        df_feat['month'] = df_feat['timestamp'].dt.month
        df_feat['day'] = df_feat['timestamp'].dt.day
        df_feat['dayofyear'] = df_feat['timestamp'].dt.dayofyear
        df_feat['is_weekend'] = df_feat['dayofweek'].isin([5, 6]).astype(int)
        df_feat['trend'] = (df_feat['timestamp'] - daily_fraud['timestamp'].min()).dt.days
        return df_feat
        
    train = create_features(daily_fraud)
    features = ['dayofweek', 'month', 'day', 'dayofyear', 'is_weekend', 'trend']
    X = train[features]
    y = train['fraud_count']
    
    # Model stabil kalması için basit ama etkili hiperparametreler
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'seed': 42,
        'verbose': -1,
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.05
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X, y)
    
    # Gelecek "days" kadar günü hazırla
    last_date = daily_fraud['timestamp'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
    future_df = pd.DataFrame({'timestamp': future_dates})
    
    future_feat = create_features(future_df)
    
    # Tahmin
    preds = model.predict(future_feat[features])
    preds = np.clip(preds, 0, None)  # Negatif tahminleri 0'a kırp
    
    future_df['predicted_fraud'] = preds
    
    # Basit güven aralığı hesaplaması (train data'daki RMSE bazlı)
    train_preds = model.predict(X)
    rmse = np.sqrt(((y - train_preds) ** 2).mean())
    margin = 1.96 * rmse
    
    future_df['upper_band'] = future_df['predicted_fraud'] + margin
    future_df['lower_band'] = np.clip(future_df['predicted_fraud'] - margin, 0, None)
    
    return daily_fraud, future_df
