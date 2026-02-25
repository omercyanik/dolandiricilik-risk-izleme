import pandas as pd
import numpy as np
import joblib
import os
import logging
from src.config import DATA_DIR, MODELS_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RiskScorer:
    def __init__(self, model_name="lgbm_calibrated.pkl"):
        model_path = MODELS_DIR / model_name
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Train the models first.")
        
        self.model = joblib.load(model_path)
        logging.info("RiskScorer Initialized. Model loaded.")

    def apply_policy(self, risk_score, allow_threshold=10, review_threshold=40):
        if risk_score >= review_threshold:
            return "block"
        elif risk_score >= allow_threshold:
            return "step_up"
        else:
            return "allow"

    def predict(self, df):
        # Assumes df has already gone through feature_engineering
        # For a full production pipeline, feature lookup (like Redis) would happen here
        probs = self.model.predict_proba(df)[:, 1]
        
        risk_scores = np.clip(probs * 100, 0, 100).round(2)
        decisions = [self.apply_policy(score) for score in risk_scores]
        
        return pd.DataFrame({
            "fraud_probability": probs.round(4),
            "risk_score": risk_scores,
            "decision": decisions
        })

if __name__ == "__main__":
    scorer = RiskScorer()
    
    test_path = DATA_DIR / "features.csv"
    if os.path.exists(test_path):
        sample_df = pd.read_csv(test_path).head(10)
        results = scorer.predict(sample_df)
        print("Sample predictions:")
        print(results)
