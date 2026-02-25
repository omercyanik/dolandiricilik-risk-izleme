import numpy as np
import pandas as pd
from typing import Dict

def simulate_roi(df_scored: pd.DataFrame, 
                 y_true_col: str = "is_fraud",
                 score_col: str = "lgbm_risk_score",
                 amount_col: str = "amount",
                 th_allow: float = 10.0,
                 th_block: float = 40.0,
                 fraud_loss_rate: float = 1.0,      # If we miss fraud, lose full amount
                 recovery_rate: float = 0.0,        # No chargeback recovery
                 review_cost_fixed: float = 5.0,    # Flat $5 cost for manual review or 3DS
                 step_up_cost_fixed: float = 0.5,   # Flat SMS/MFA challenge cost
                 fp_margin_loss: float = 0.05       # 5% margin loss on declined good transactions
                 ) -> Dict[str, float]:
    """
    Computes business ROI for a specific threshold policy.
    """
    
    # Decisions based on score
    conditions = [
        df_scored[score_col] >= th_block,
        (df_scored[score_col] >= th_allow) & (df_scored[score_col] < th_block)
    ]
    choices = ["block", "step_up"]
    df_scored["action"] = np.select(conditions, choices, default="allow")
    
    total_tx = len(df_scored)
    total_amount = df_scored[amount_col].sum()
    
    # Baseline: No ML, accept everything. Loss is all fraud amount.
    fraud_amount = df_scored.loc[df_scored[y_true_col] == 1, amount_col].sum()
    baseline_loss = fraud_amount * (1.0 - recovery_rate)
    
    # Policy Impact
    # 1. Blocked transactions
    blocked = df_scored[df_scored["action"] == "block"]
    tps_blocked = blocked[blocked[y_true_col] == 1]
    fps_blocked = blocked[blocked[y_true_col] == 0]
    
    fraud_prevented_amount = tps_blocked[amount_col].sum()
    fp_loss = fps_blocked[amount_col].sum() * fp_margin_loss
    manual_review_cost = len(blocked) * review_cost_fixed # Assuming all blocks are reviewed
    
    # 2. Step-up transactions
    step_up = df_scored[df_scored["action"] == "step_up"]
    # Simplification: Step-up succeeds in blocking 80% of fraud, but 5% good customers abandon
    tps_stepped_up = step_up[step_up[y_true_col] == 1]
    fps_stepped_up = step_up[step_up[y_true_col] == 0]
    
    step_up_cost = len(step_up) * step_up_cost_fixed
    step_up_fraud_prevented = tps_stepped_up[amount_col].sum() * 0.80
    step_up_fp_abandon_loss = fps_stepped_up[amount_col].sum() * 0.05 * fp_margin_loss
    
    # Missed fraud (allowed or slipped through step-up)
    allowed = df_scored[df_scored["action"] == "allow"]
    missed_fraud_amount = allowed.loc[allowed[y_true_col] == 1, amount_col].sum()
    missed_fraud_amount += tps_stepped_up[amount_col].sum() * 0.20
    
    # Total new losses
    new_fraud_loss = missed_fraud_amount * (1.0 - recovery_rate)
    operational_cost = manual_review_cost + step_up_cost + fp_loss + step_up_fp_abandon_loss
    total_policy_cost = new_fraud_loss + operational_cost
    
    savings = baseline_loss - total_policy_cost
    roi_percent = (savings / max(operational_cost, 1.0)) * 100
    
    capture_rate = (fraud_prevented_amount + step_up_fraud_prevented) / max(fraud_amount, 1.0)
    review_rate = len(blocked) / float(total_tx)
    step_up_rate = len(step_up) / float(total_tx)

    return {
        "Total Count": total_tx,
        "Total Amount": float(total_amount),
        "Baseline Fraud Loss": float(baseline_loss),
        "New Fraud Loss": float(new_fraud_loss),
        "Operational Cost": float(operational_cost),
        "Total Policy Cost": float(total_policy_cost),
        "Gross Savings": float(savings),
        "ROI %": float(roi_percent),
        "Fraud Capture Rate": float(capture_rate),
        "Review Rate": float(review_rate),
        "Step-Up Rate": float(step_up_rate)
    }

def find_optimal_threshold(df_scored, y_true_col="is_fraud", score_col="lgbm_risk_score"):
    # Grid search for threshold combination
    best_savings = -float("inf")
    best_th_allow = 10.0
    best_th_block = 40.0
    
    # Keep it simple, just moving block threshold and keeping a 10pt buffer.
    for block_th in range(20, 95, 5):
        allow_th = max(5, block_th - 15)
        metrics = simulate_roi(df_scored, y_true_col, score_col, th_allow=allow_th, th_block=block_th)
        if metrics["Gross Savings"] > best_savings:
            best_savings = metrics["Gross Savings"]
            best_th_allow = allow_th
            best_th_block = block_th
            
    return best_th_allow, best_th_block, best_savings
