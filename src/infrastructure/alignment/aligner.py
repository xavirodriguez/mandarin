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
                if any(
                    '\u4e00' <= c <= '\u9fff' or '\u3400' <= c <= '\u4dbf' or '\uf900' <= c <= '\ufaff'
                    for c in token
                ):
                    pinyin_lists = pypinyin.lazy_pinyin(token, style=pypinyin.Style.NORMAL, heteronym=True)
                    for py_group in pinyin_lists:
                        for py in py_group:
                            py_clean = self._strip_tone_marks(py).lower().replace("ü", "v")
                            pinyin_map.setdefault(py_clean, []).append(tid)

                clean_tok = token.replace("##", "").replace("<", "").replace(">", "").replace("|", "").strip().lower()
                clean_tok = self._strip_tone_marks(clean_tok).replace("ü", "v")
                if clean_tok and clean_tok.isalpha():
                    pinyin_map.setdefault(clean_tok, []).append(tid)

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

            target_token_sets = []
            for py in clean_targets:
                py_clean = py.lower().replace("ü", "v")
                tids = pinyin_map.get(py_clean, [])
                if not tids:
                    onset_str, final_str = self._split_pinyin_components(py_clean)
                    tids = pinyin_map.get(onset_str, []) + pinyin_map.get(final_str, [])
                if not tids:
                    tids = list(range(1, min(50, posteriors.shape[-1])))
                target_token_sets.append(list(set(tids)))

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
            min_syllable_frames = max(1, num_frames // (num_units * 2))

            for i, py in enumerate(pinyin_units):
                frame_indices = np.where(path == i)[0]
                if len(frame_indices) >= min_syllable_frames:
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

                start_sample = int(s_start * sample_rate)
                end_sample = int(s_end * sample_rate)
                syl_audio = audio[start_sample:end_sample] if (audio is not None and len(audio) > end_sample) else audio

                onset, nucleus, coda = self._decompose_pinyin_with_features(py, s_start, s_end, syl_audio, sample_rate)

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

    def _split_pinyin_components(self, pinyin: str) -> Tuple[str, str]:
        cleaned = self._strip_tone_marks(pinyin).lower().replace("ü", "v")
        if len(cleaned) > 1 and cleaned[:2] in ("zh", "ch", "sh"):
            return cleaned[:2], cleaned[2:]
        elif len(cleaned) > 0 and cleaned[0] not in "aeiou":
            return cleaned[0], cleaned[1:]
        return "", cleaned

    def _decompose_pinyin_with_features(
        self,
        pinyin: str,
        s_start: float,
        s_end: float,
        syl_audio: Optional[np.ndarray],
        sample_rate: int
    ) -> Tuple[Optional[PhonemeSegment], PhonemeSegment, Optional[PhonemeSegment]]:
        dur = max(0.01, s_end - s_start)
        onset_str, final_str = self._split_pinyin_components(pinyin)

        onset_ratio = 0.25 if onset_str else 0.0
        coda_ratio = 0.25 if (final_str.endswith("n") or final_str.endswith("ng")) else 0.0

        if syl_audio is not None and len(syl_audio) > 160:
            frame_len = 160
            n_frames = len(syl_audio) // frame_len
            if n_frames >= 4:
                energies = np.array([np.mean(syl_audio[k*frame_len:(k+1)*frame_len]**2) for k in range(n_frames)])
                max_e = np.max(energies) + 1e-8
                norm_e = energies / max_e

                if onset_str:
                    high_e_idx = np.where(norm_e > 0.3)[0]
                    onset_frame = high_e_idx[0] if len(high_e_idx) > 0 else int(n_frames * 0.25)
                    onset_ratio = float(np.clip(onset_frame / float(n_frames), 0.15, 0.40))

                if coda_ratio > 0:
                    low_e_idx = np.where(norm_e > 0.4)[0]
                    coda_frame = low_e_idx[-1] + 1 if len(low_e_idx) > 0 else int(n_frames * 0.75)
                    coda_ratio = float(np.clip((n_frames - coda_frame) / float(n_frames), 0.15, 0.40))

        if onset_str:
            onset_end = s_start + dur * onset_ratio
            onset_seg = PhonemeSegment(phoneme=onset_str, start=s_start, end=onset_end, confidence=0.95)
            nuc_start = onset_end
        else:
            onset_seg = None
            nuc_start = s_start

        if final_str.endswith("n") or final_str.endswith("ng"):
            nuc_str = final_str[:-2] if final_str.endswith("ng") else final_str[:-1]
            coda_str = "ng" if final_str.endswith("ng") else "n"

            coda_start = max(nuc_start + 0.01, s_end - dur * coda_ratio)
            nucleus_seg = PhonemeSegment(phoneme=nuc_str or final_str, start=nuc_start, end=coda_start, confidence=0.95)
            coda_seg = PhonemeSegment(phoneme=coda_str, start=coda_start, end=s_end, confidence=0.95)
        else:
            nucleus_seg = PhonemeSegment(phoneme=final_str or pinyin, start=nuc_start, end=s_end, confidence=0.95)
            coda_seg = None

        return onset_seg, nucleus_seg, coda_seg

    def _decompose_pinyin(self, pinyin: str, s_start: float, s_end: float) -> Tuple[Optional[PhonemeSegment], PhonemeSegment, Optional[PhonemeSegment]]:
        return self._decompose_pinyin_with_features(pinyin, s_start, s_end, None, 16000)
