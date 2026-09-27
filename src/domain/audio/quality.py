from dataclasses import dataclass, field
from typing import List

@dataclass(frozen=True)
class AudioQualityMetrics:
    snr_db: float
    clipping_ratio: float
    speech_duration_sec: float
    total_duration_sec: float
    voiced_ratio: float
    is_acceptable: bool
    rejection_reasons: List[str] = field(default_factory=list)
    quality_score: float = 1.0  # 0.0 to 1.0

    def should_reject(self) -> bool:
        return not self.is_acceptable
