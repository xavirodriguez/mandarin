from dataclasses import dataclass, field
from typing import Optional, List, Dict
from src.domain.phonetics.models import PhoneticAssessmentResult, PhonemeSegment
from src.domain.tones.models import ToneAssessmentResult
from src.domain.audio.quality import AudioQualityMetrics

@dataclass
class SyllableStructure:
    onset: Optional[PhonemeSegment] = None
    nucleus: Optional[PhonemeSegment] = None
    coda: Optional[PhonemeSegment] = None

@dataclass
class SyllableAssessment:
    syllable: str              # e.g., "ma"
    pinyin: str                # e.g., "mā"
    start: float
    end: float
    duration: float

    phonetics: PhoneticAssessmentResult
    tone: ToneAssessmentResult
    structure: SyllableStructure
    audio_quality: AudioQualityMetrics

    @property
    def overall_score(self) -> float:
        # Weighted combination of phonetic score and tone similarity
        phonetic_weight = 0.5
        tone_weight = 0.5
        tone_score = self.tone.tone_probabilities.get(self.tone.contextual_target, 0.0)
        return float(phonetic_weight * self.phonetics.overall_score + tone_weight * tone_score)
