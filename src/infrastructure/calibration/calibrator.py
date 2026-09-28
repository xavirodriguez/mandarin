import numpy as np
from typing import Dict, Tuple

class ModelCalibrator:
    """
    Uncertainty-aware calibration module implementing:
    - Temperature Scaling for logits/probabilities
    - Expected Calibration Error (ECE) metric calculation
    - Insufficient evidence / low confidence gating
    """

    def __init__(self, temperature: float = 1.2, min_confidence_threshold: float = 0.50):
        self.temperature = max(0.1, temperature)
        self.min_confidence_threshold = min_confidence_threshold

    def calibrate_probabilities(self, probs: Dict[int, float]) -> Dict[int, float]:
        """
        Applies temperature scaling to soften/calibrate predicted class probabilities.
        P_i = exp(log(p_i) / T) / sum_j exp(log(p_j) / T)
        """
        keys = list(probs.keys())
        raw_p = np.array([max(1e-6, probs[k]) for k in keys], dtype=np.float32)
        logits = np.log(raw_p)

        scaled_logits = logits / self.temperature
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        calibrated_probs = exp_logits / np.sum(exp_logits)

        return {keys[i]: float(calibrated_probs[i]) for i in range(len(keys))}

    def calculate_ece(self, confidences: np.ndarray, accuracies: np.ndarray, n_bins: int = 10) -> float:
        """
        Computes Expected Calibration Error (ECE):
        ECE = sum_{b=1}^B (|B_b| / N) * |acc(B_b) - conf(B_b)|
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n_samples = len(confidences)
        if n_samples == 0:
            return 0.0

        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            prop_in_bin = np.mean(in_bin)
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

        return float(ece)

    def is_evidence_sufficient(self, confidence: float, quality_score: float) -> bool:
        """
        Checks if model confidence and audio quality are above minimal reliability thresholds.
        """
        return confidence >= self.min_confidence_threshold and quality_score >= 0.30
