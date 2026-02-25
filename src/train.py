import pandas as pd
import numpy as np
import logging
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import optuna
from datetime import timedelta

from src.config import DATA_DIR, MODELS_DIR, TARGET_COL, NUMERIC_FEATURES, CATEGORICAL_FEATURES

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_metrics(y_true, y_prob, review_rate=0.01):
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    
    # Calculate recall at fixed review rate
    threshold_idx = int(len(y_prob) * (1 - review_rate))
    if threshold_idx >= len(y_prob):
        threshold_idx = len(y_prob) - 1
    
    sort_idx = np.argsort(y_prob)
    threshold = y_prob[sort_idx[threshold_idx]]
    
    y_pred = (y_prob >= threshold).astype(int)
    recall_at_rr = recall_score(y_true, y_pred, zero_division=0)
    precision_at_rr = precision_score(y_true, y_pred, zero_division=0)
    brier = brier_score_loss(y_true, y_prob)
    
    return {
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        f"Recall @ {review_rate*100}%": recall_at_rr,
        f"Precision @ {review_rate*100}%": precision_at_rr,
        "Brier Score": brier
    }

def train_models():
    logging.info("loading features...")
    df = pd.read_csv(DATA_DIR / "features.csv", parse_dates=['timestamp'])
    
    # Time-based split: Use the last 20% of days for testing
    max_time = df['timestamp'].max()
    min_time = df['timestamp'].min()
    duration = max_time - min_time
    test_start = max_time - timedelta(days=int(duration.days * 0.2))
    
    train = df[df['timestamp'] < test_start].copy()
    test = df[df['timestamp'] >= test_start].copy()
    
    logging.info(f"train: {len(train)} | test: {len(test)}")
    logging.info(f"fraud rate -> train: {train[TARGET_COL].mean():.4f}, test: {test[TARGET_COL].mean():.4f}")
    
    # Preprocessing
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ])
    
    X_train = train[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train[TARGET_COL]
    X_test = test[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test = test[TARGET_COL]
    
    # --- baseline LR
    logging.info("training baseline LR...")
    lr = Pipeline(steps=[('preprocessor', preprocessor),
                         ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))])
    
    lr.fit(X_train, y_train)
    lr_probs = lr.predict_proba(X_test)[:, 1]
    
    # --- train LGBM
    logging.info("training LGBM...")
    # scale pos weight for imbalance
    scale_pos_weight = (len(y_train) - sum(y_train)) / max(sum(y_train), 1)
    
    
    # --- optuna tuning
    logging.info("running optuna optimization...")
    
    # holdout for optuna
    X_train_optuna, X_val_optuna, y_train_optuna, y_val_optuna = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    X_t_pre = preprocessor.fit_transform(X_train_optuna)
    X_v_pre = preprocessor.transform(X_val_optuna)
    
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100),
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        gbm = lgb.LGBMClassifier(**param)
        gbm.fit(X_t_pre, y_train_optuna)
        y_pred = gbm.predict_proba(X_v_pre)[:, 1]
        
        return average_precision_score(y_val_optuna, y_pred)
    
    # maximizing PR-AUC for minority class
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    
    logging.info(f"best params: {study.best_params}")
    logging.info(f"best PR-AUC: {study.best_value:.4f}")
    
    best_params = study.best_params
    best_params['scale_pos_weight'] = scale_pos_weight
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1
    best_params['verbose'] = -1
    
    logging.info("training final model...")
    lgbm_model = lgb.LGBMClassifier(**best_params)
    
    lgbm = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', lgbm_model)])
    lgbm.fit(X_train, y_train)
    
    # skipping calibration due to extreme class imbalance wiping out minority probabilities.
    # using raw scores directly.
    logging.info("extracting raw probabilities...")
    
    lgbm_probs = lgbm.predict_proba(X_test)[:, 1]
    
    # eval
    logging.info("--- baseline LR ---")
    metrics_lr = calculate_metrics(y_test, lr_probs)
    for k, v in metrics_lr.items(): logging.info(f"{k}: {v:.4f}")
        
    logging.info("--- final LGBM ---")
    metrics_lgbm = calculate_metrics(y_test, lgbm_probs)
    for k, v in metrics_lgbm.items(): logging.info(f"{k}: {v:.4f}")
        
    # Save test dataset and probabilities for predict/roi
    test['lgbm_risk_prob'] = lgbm_probs
    test['lgbm_risk_score'] = (lgbm_probs * 100).clip(0, 100)
    test['timestamp'] = pd.to_datetime(test['timestamp']).dt.strftime('%d-%m-%Y %H:%M:%S')
    test.to_csv(DATA_DIR / "scored_test_set.csv", index=False)
    
    # save models
    joblib.dump(lr, MODELS_DIR / "lr_baseline.pkl")
    joblib.dump(lgbm, MODELS_DIR / "lgbm_calibrated.pkl") # (uncalibrated, kept name for compat)
    logging.info(f"done. models saved to {MODELS_DIR}")

if __name__ == "__main__":
    train_models()
