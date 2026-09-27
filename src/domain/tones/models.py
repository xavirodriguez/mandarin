from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np

@dataclass
class PitchContour:
    time_stamps: np.ndarray  # float array
    f0_values: np.ndarray    # Hz array, 0 for unvoiced
    voiced_probs: np.ndarray # 0.0 to 1.0
    normalized_f0: Optional[np.ndarray] = None # Speaker normalized log-F0 z-score or semitones

    @property
    def mean_f0(self) -> float:
        voiced = self.f0_values[self.voiced_probs > 0.5]
        return float(np.mean(voiced)) if len(voiced) > 0 else 0.0

    @property
    def slope(self) -> float:
        voiced_idx = np.where(self.voiced_probs > 0.5)[0]
        if len(voiced_idx) < 2:
            return 0.0
        y = self.f0_values[voiced_idx]
        x = self.time_stamps[voiced_idx]
        return float((y[-1] - y[0]) / (x[-1] - x[0] + 1e-6))

    @property
    def range_hz(self) -> float:
        voiced = self.f0_values[self.voiced_probs > 0.5]
        if len(voiced) == 0:
            return 0.0
        return float(np.ptp(voiced))

@dataclass(frozen=True)
class ToneAssessmentResult:
    lexical_tone: int          # 1, 2, 3, 4, 0 (neutral)
    contextual_target: int     # Target after tone sandhi
    predicted_tone: int        # Predicted tone class
    tone_probabilities: Dict[int, float] # Distribution over tones {1: 0.8, 2: 0.1, ...}
    classification_confidence: float
    contour_similarity: float  # DTW or correlation score (0 to 1)
    pitch_contour: PitchContour
    is_sandhi_applied: bool = False
    sandhi_rule_name: Optional[str] = None
