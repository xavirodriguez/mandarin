import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

@dataclass
class CAPTDatasetSample:
    speaker_id: str
    utterance_id: str
    target_text: str
    pinyin: List[str]
    phoneme_sequence: List[str]
    syllable_boundaries: List[Dict[str, float]] # [{'start': 0.0, 'end': 0.5}, ...]
    lexical_tones: List[int]
    contextual_tones: List[int]
    observed_pronunciation: List[str]
    error_type: Optional[str] = None
    error_severity: float = 0.0
    audio_path: Optional[str] = None

class CAPTDatasetLoader:
    """
    Dataset loader enforcing strict Speaker-Independent train/val/test splits to avoid data leakage.
    """

    def __init__(self, samples: List[CAPTDatasetSample]):
        self.samples = samples

    def get_speaker_independent_splits(
        self, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15
    ) -> Tuple[List[CAPTDatasetSample], List[CAPTDatasetSample], List[CAPTDatasetSample]]:
        unique_speakers = sorted(list(set(s.speaker_id for s in self.samples)))
        np.random.seed(42)
        np.random.shuffle(unique_speakers)

        n_speakers = len(unique_speakers)
        n_train = int(n_speakers * train_ratio)
        n_val = int(n_speakers * val_ratio)

        train_speakers = set(unique_speakers[:n_train])
        val_speakers = set(unique_speakers[n_train:n_train + n_val])
        test_speakers = set(unique_speakers[n_train + n_val:])

        train_samples = [s for s in self.samples if s.speaker_id in train_speakers]
        val_samples = [s for s in self.samples if s.speaker_id in val_speakers]
        test_samples = [s for s in self.samples if s.speaker_id in test_speakers]

        return train_samples, val_samples, test_samples
