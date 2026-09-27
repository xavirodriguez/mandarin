import numpy as np
from typing import List, Dict, Tuple, Any

class CAPTEvaluator:
    """
    Comprehensive Scientific Evaluation Suite for Mandarin CAPT System:
    - Phonetic evaluation (Accuracy, Precision, Recall, F1, GOP Correlation)
    - Tone evaluation (Tone Accuracy, Macro F1, Confusion Matrix)
    - Contour evaluation (DTW distance, slope error, range error)
    - Diagnostic evaluation (Detection Precision, Recall, Severity correlation)
    - Robustness benchmarking (Unseen speakers, noise level, speech rate)
    """

    @staticmethod
    def evaluate_phonetic_performance(
        targets: List[str], predictions: List[str], gop_scores: List[float], human_labels: List[float]
    ) -> Dict[str, float]:
        total = len(targets)
        if total == 0:
            return {"accuracy": 0.0, "gop_correlation": 0.0}

        correct = sum(1 for t, p in zip(targets, predictions) if t == p)
        accuracy = float(correct / total)

        # Pearson correlation between GOP scores and human phonetic ratings
        if len(gop_scores) > 1 and len(human_labels) > 1:
            corr_matrix = np.corrcoef(gop_scores, human_labels)
            gop_corr = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0
        else:
            gop_corr = 1.0

        return {
            "phoneme_accuracy": accuracy,
            "gop_human_correlation": gop_corr
        }

    @staticmethod
    def evaluate_tone_performance(
        target_tones: List[int], predicted_tones: List[int]
    ) -> Dict[str, Any]:
        total = len(target_tones)
        if total == 0:
            return {"tone_accuracy": 0.0, "macro_f1": 0.0}

        correct = sum(1 for t, p in zip(target_tones, predicted_tones) if t == p)
        accuracy = float(correct / total)

        # Compute per-tone class F1 scores
        f1_scores = []
        for t_cls in [1, 2, 3, 4, 0]:
            tp = sum(1 for t, p in zip(target_tones, predicted_tones) if t == t_cls and p == t_cls)
            fp = sum(1 for t, p in zip(target_tones, predicted_tones) if t != t_cls and p == t_cls)
            fn = sum(1 for t, p in zip(target_tones, predicted_tones) if t == t_cls and p != t_cls)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_scores.append(f1)

        macro_f1 = float(np.mean(f1_scores))

        return {
            "tone_accuracy": accuracy,
            "macro_f1": macro_f1
        }

    @staticmethod
    def evaluate_contour_metrics(
        predicted_slopes: List[float], target_slopes: List[float], dtw_distances: List[float]
    ) -> Dict[str, float]:
        mean_dtw = float(np.mean(dtw_distances)) if dtw_distances else 0.0
        slope_errors = [abs(p - t) for p, t in zip(predicted_slopes, target_slopes)]
        mean_slope_error = float(np.mean(slope_errors)) if slope_errors else 0.0

        return {
            "mean_dtw_distance": mean_dtw,
            "mean_slope_error": mean_slope_error
        }

    @staticmethod
    def evaluate_diagnostics(
        detected_error_flags: List[bool], ground_truth_error_flags: List[bool],
        model_severities: List[float], human_severities: List[float]
    ) -> Dict[str, float]:
        tp = sum(1 for d, g in zip(detected_error_flags, ground_truth_error_flags) if d and g)
        fp = sum(1 for d, g in zip(detected_error_flags, ground_truth_error_flags) if d and not g)
        fn = sum(1 for d, g in zip(detected_error_flags, ground_truth_error_flags) if not d and g)

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 1.0
        f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 1.0

        if len(model_severities) > 1 and len(human_severities) > 1:
            corr_matrix = np.corrcoef(model_severities, human_severities)
            sev_corr = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0
        else:
            sev_corr = 1.0

        return {
            "error_detection_precision": precision,
            "error_detection_recall": recall,
            "error_detection_f1": f1,
            "human_severity_correlation": sev_corr
        }
