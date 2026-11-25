"""Continuous monitoring and drift alerting daemon simulation.
Authored by Khushi Jangra (Data Lead & Drift Monitoring).
"""

from pathlib import Path
import json
import logging
import pandas as pd
from src.solar.clean import clean_features, load_and_clean_train
from src.solar.monitoring import compute_psi

logger = logging.getLogger("drift_monitor")
logger.setLevel(logging.INFO)


def evaluate_feature_drift(
    reference_path: str = "data/raw/train.csv",
    incoming_path: str = "data/raw/later.csv",
    psi_alert_threshold: float = 0.25
) -> dict:
    """Analyze incoming batches against baseline reference and trigger alerts if PSI exceeds threshold."""
    ref_df = load_and_clean_train(reference_path)
    inc_df = pd.read_csv(incoming_path)
    
    numeric_features = [
        "panel_area_m2", "sunshine_hours", "temp_c", 
        "dust_days", "panel_age_yr", "cloud_cover_pct"
    ]
    
    drift_report = {}
    alerts = []
    
    for col in numeric_features:
        psi_val = compute_psi(ref_df[col].values, inc_df[col].values, num_bins=10)
        status = "CRITICAL_DRIFT" if psi_val >= psi_alert_threshold else ("MODERATE_DRIFT" if psi_val >= 0.1 else "STABLE")
        drift_report[col] = {"psi": psi_val, "status": status}
        if psi_val >= psi_alert_threshold:
            alerts.append({"feature": col, "psi": psi_val, "action": "Trigger Retraining Pipeline"})
            
    return {
        "status": "ALERT" if alerts else "HEALTHY",
        "alerts": alerts,
        "features": drift_report
    }


if __name__ == "__main__":
    report = evaluate_feature_drift()
    print("Drift Monitoring Assessment:\n", json.dumps(report, indent=2))
