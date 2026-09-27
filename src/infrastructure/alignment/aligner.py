import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from src.domain.phonetics.models import PhonemeSegment

class forced_aligner_mock:
    """
    Mock/Default Forced Aligner implementing temporal alignment for Syllables into Onset, Nucleus, Coda.
    """

    @staticmethod
    def _strip_tone_marks(text: str) -> str:
        trans_table = str.maketrans(
            "āáǎàōóǒòēéěèīíǐìūúǔùǖǘǚǜ12340",
            "aaaaeeeeiiiioooouuuuuuuu     "
        )
        return text.translate(trans_table).replace(" ", "")

    def align(self, audio: np.ndarray, sample_rate: int, pinyin_units: List[str]) -> List[Dict[str, Any]]:
        total_duration = len(audio) / float(sample_rate)
        num_units = len(pinyin_units)
        if num_units == 0:
            return []

        syllable_dur = total_duration / num_units
        alignments = []

        for i, py in enumerate(pinyin_units):
            s_start = i * syllable_dur
            s_end = (i + 1) * syllable_dur

            # Decompose into onset, nucleus, coda
            onset, nucleus, coda = self._decompose_pinyin(py, s_start, s_end)

            alignments.append({
                "pinyin": py,
                "start": float(s_start),
                "end": float(s_end),
                "confidence": 0.95,
                "onset": onset,
                "nucleus": nucleus,
                "coda": coda
            })

        return alignments

    def _decompose_pinyin(self, pinyin: str, s_start: float, s_end: float) -> Tuple[Optional[PhonemeSegment], PhonemeSegment, Optional[PhonemeSegment]]:
        dur = s_end - s_start
        cleaned = self._strip_tone_marks(pinyin)

        if len(cleaned) > 1 and cleaned[:2] in ("zh", "ch", "sh"):
            onset_str = cleaned[:2]
            final_str = cleaned[2:]
        elif len(cleaned) > 0 and cleaned[0] not in "aeiou":
            onset_str = cleaned[0]
            final_str = cleaned[1:]
        else:
            onset_str = ""
            final_str = cleaned

        if onset_str:
            onset_seg = PhonemeSegment(phoneme=onset_str, start=s_start, end=s_start + dur * 0.3, confidence=0.95)
            nuc_start = s_start + dur * 0.3
        else:
            onset_seg = None
            nuc_start = s_start

        if final_str.endswith("n") or final_str.endswith("ng"):
            nuc_str = final_str[:-2] if final_str.endswith("ng") else final_str[:-1]
            coda_str = "ng" if final_str.endswith("ng") else "n"

            nuc_end = nuc_start + dur * 0.5
            nucleus_seg = PhonemeSegment(phoneme=nuc_str or final_str, start=nuc_start, end=nuc_end, confidence=0.95)
            coda_seg = PhonemeSegment(phoneme=coda_str, start=nuc_end, end=s_end, confidence=0.95)
        else:
            nucleus_seg = PhonemeSegment(phoneme=final_str or cleaned, start=nuc_start, end=s_end, confidence=0.95)
            coda_seg = None

        return onset_seg, nucleus_seg, coda_seg
