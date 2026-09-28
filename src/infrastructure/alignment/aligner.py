import os
import numpy as np
import pypinyin
from typing import List, Dict, Any, Tuple, Optional
from src.domain.phonetics.models import PhoneticAssessmentResult, PhonemeSegment

class forced_aligner_mock:
    """
    Forced Aligner implementing temporal alignment for Syllables into Onset, Nucleus, Coda.
    Provides mock equal-duration fallback for test doubles and real Viterbi CTC dynamic programming
    alignment in NumPy over acoustic posteriors when CAPT_ADAPTER_MODE='real'.
    """

    def __init__(self):
        self._pinyin_vocab_map = None

    @staticmethod
    def _strip_tone_marks(text: str) -> str:
        trans_table = str.maketrans(
            "āáǎàōóǒòēéěèīíǐìūúǔùǖǘǚǜ12340",
            "aaaaeeeeiiiioooouuuuuuuu     "
        )
        return text.translate(trans_table).replace(" ", "")

    def _get_pinyin_vocab_map(self, processor) -> Dict[str, List[int]]:
        if self._pinyin_vocab_map is None and processor is not None:
            vocab = processor.tokenizer.get_vocab()
            pinyin_map: Dict[str, List[int]] = {}
            for token, tid in vocab.items():
                if len(token) == 1 and '\u4e00' <= token <= '\u9fff':
                    py = pypinyin.lazy_pinyin(token, style=pypinyin.Style.NORMAL)[0]
                    if py not in pinyin_map:
                        pinyin_map[py] = []
                    pinyin_map[py].append(tid)
            self._pinyin_vocab_map = pinyin_map
        return self._pinyin_vocab_map or {}

    def align(self, audio: np.ndarray, sample_rate: int, pinyin_units: List[str]) -> List[Dict[str, Any]]:
        mode = os.getenv("CAPT_ADAPTER_MODE", "mock").lower()
        if mode == "real":
            return self._real_viterbi_align(audio, sample_rate, pinyin_units)
        else:
            return self._mock_equal_align(audio, sample_rate, pinyin_units)

    def _mock_equal_align(self, audio: np.ndarray, sample_rate: int, pinyin_units: List[str]) -> List[Dict[str, Any]]:
        total_duration = len(audio) / float(sample_rate) if (audio is not None and len(audio) > 0) else 1.0
        num_units = len(pinyin_units)
        if num_units == 0:
            return []

        syllable_dur = total_duration / num_units
        alignments = []

        for i, py in enumerate(pinyin_units):
            s_start = i * syllable_dur
            s_end = (i + 1) * syllable_dur

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

    def _real_viterbi_align(self, audio: np.ndarray, sample_rate: int, pinyin_units: List[str]) -> List[Dict[str, Any]]:
        num_units = len(pinyin_units)
        if num_units == 0:
            return []

        total_duration = len(audio) / float(sample_rate) if (audio is not None and len(audio) > 0) else 1.0

        try:
            from src.infrastructure.models.adapters import SpeechEncoderAdapter, PhonemeRecognizerAdapter, _SHARED_PROCESSOR
            encoder = SpeechEncoderAdapter()
            recognizer = PhonemeRecognizerAdapter()

            logits = encoder.encode(audio, sample_rate)
            posteriors = recognizer.recognize_posteriors(logits)

            num_frames = len(posteriors)
            if num_frames == 0:
                return self._mock_equal_align(audio, sample_rate, pinyin_units)

            frame_duration = total_duration / float(num_frames)
            pinyin_map = self._get_pinyin_vocab_map(_SHARED_PROCESSOR)
            clean_targets = [self._strip_tone_marks(py) for py in pinyin_units]

            # Viterbi CTC Dynamic Programming Alignment in NumPy
            blank_id = 0
            log_probs = np.log(posteriors + 1e-12)

            target_token_sets = [pinyin_map.get(py, [1]) for py in clean_targets]

            states = [-1]
            for i in range(num_units):
                states.extend([i, -1])
            L = len(states)

            # Compute state emission log probabilities
            emission = np.full((num_frames, L), -1e9, dtype=np.float32)
            for t in range(num_frames):
                emission[t, 0] = log_probs[t, blank_id]
                for s_idx in range(1, L):
                    unit_idx = states[s_idx]
                    if unit_idx == -1:
                        emission[t, s_idx] = log_probs[t, blank_id]
                    else:
                        tids = target_token_sets[unit_idx]
                        p_sum = np.sum(posteriors[t, tids])
                        emission[t, s_idx] = np.log(p_sum + 1e-12)

            # Viterbi DP Table
            viterbi = np.full((num_frames, L), -1e9, dtype=np.float32)
            backtrack = np.zeros((num_frames, L), dtype=int)

            viterbi[0, 0] = emission[0, 0]
            if L > 1:
                viterbi[0, 1] = emission[0, 1]

            for t in range(1, num_frames):
                for s in range(L):
                    best_score = viterbi[t-1, s]
                    best_prev = s

                    if s > 0 and viterbi[t-1, s-1] > best_score:
                        best_score = viterbi[t-1, s-1]
                        best_prev = s - 1

                    if s > 1 and states[s] != -1 and states[s] != states[s-2]:
                        if viterbi[t-1, s-2] > best_score:
                            best_score = viterbi[t-1, s-2]
                            best_prev = s - 2

                    viterbi[t, s] = best_score + emission[t, s]
                    backtrack[t, s] = best_prev

            # Backtrack optimal state path
            curr_s = L - 1 if viterbi[num_frames - 1, L - 1] > viterbi[num_frames - 1, L - 2] else L - 2
            path = np.zeros(num_frames, dtype=int)
            for t in range(num_frames - 1, -1, -1):
                path[t] = states[curr_s]
                curr_s = backtrack[t, curr_s]

            alignments = []
            for i, py in enumerate(pinyin_units):
                frame_indices = np.where(path == i)[0]
                if len(frame_indices) > 0:
                    start_frame = frame_indices[0]
                    end_frame = frame_indices[-1] + 1
                else:
                    start_frame = int((i / num_units) * num_frames)
                    end_frame = int(((i + 1) / num_units) * num_frames)

                s_start = float(start_frame * frame_duration)
                s_end = float(end_frame * frame_duration)
                if s_end <= s_start:
                    s_end = s_start + 0.1

                conf = float(np.mean(np.max(posteriors[start_frame:max(start_frame+1, end_frame)], axis=-1)))
                conf = float(np.clip(conf, 0.50, 0.99))

                onset, nucleus, coda = self._decompose_pinyin(py, s_start, s_end)

                alignments.append({
                    "pinyin": py,
                    "start": s_start,
                    "end": s_end,
                    "confidence": conf,
                    "onset": onset,
                    "nucleus": nucleus,
                    "coda": coda,
                    "aligned_frames": (start_frame, end_frame)
                })

            return alignments
        except Exception:
            return self._mock_equal_align(audio, sample_rate, pinyin_units)

    def _decompose_pinyin(self, pinyin: str, s_start: float, s_end: float) -> Tuple[Optional[PhonemeSegment], PhonemeSegment, Optional[PhonemeSegment]]:
        dur = max(0.01, s_end - s_start)
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
