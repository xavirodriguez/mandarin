from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict

class PhonemeCategory(Enum):
    INITIAL = "initial"
    FINAL = "final"
    TONE = "tone"

# Mandarin confusion pairs
MANDARIN_CONFUSION_PAIRS = {
    ("zh", "z"), ("z", "zh"),
    ("ch", "c"), ("c", "ch"),
    ("sh", "s"), ("s", "sh"),
    ("j", "q"),  ("q", "j"),
    ("q", "x"),  ("x", "q"),
    ("n", "l"),  ("l", "n"),
    ("f", "h"),  ("h", "f"),
    ("b", "p"),  ("p", "b"),
    ("d", "t"),  ("t", "d"),
    ("g", "k"),  ("k", "g")
}

@dataclass(frozen=True)
class PhonemeSegment:
    phoneme: str
    start: float
    end: float
    confidence: float

@dataclass(frozen=True)
class PhoneticAssessmentResult:
    target_phonemes: List[str]
    detected_phonemes: List[str]
    phoneme_scores: Dict[str, float]  # GOP or similarity per phoneme
    overall_score: float
    confidence: float
    substitutions: List[tuple[str, str]] = field(default_factory=list) # (target, detected)
    omissions: List[str] = field(default_factory=list)
    insertions: List[str] = field(default_factory=list)
    confusion_type: Optional[str] = None  # e.g., "retroflex_vs_dental", "aspirated_vs_unaspirated"
