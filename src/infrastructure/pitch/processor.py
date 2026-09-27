import numpy as np
from typing import Tuple, Optional
from src.domain.tones.models import PitchContour

class PitchProcessor:
    """
    Robust Pitch Estimator and Speaker Normalizer.
    Estimates F0 (YIN/PYIN style or autocorrelation), voiced probability,
    and applies Log-F0 Z-score / Semitone speaker normalization.
    """

    def __init__(self, sample_rate: int = 16000, frame_length_ms: float = 25.0, hop_length_ms: float = 10.0):
        self.sample_rate = sample_rate
        self.frame_len = int(sample_rate * frame_length_ms / 1000.0)
        self.hop_len = int(sample_rate * hop_length_ms / 1000.0)

    def estimate_pitch(self, audio: np.ndarray) -> PitchContour:
        """
        Autocorrelation-based pitch estimation with voiced/unvoiced decision.
        """
        n_frames = max(1, (len(audio) - self.frame_len) // self.hop_len + 1)
        time_stamps = np.array([i * self.hop_len / self.sample_rate for i in range(n_frames)], dtype=np.float32)
        f0_values = np.zeros(n_frames, dtype=np.float32)
        voiced_probs = np.zeros(n_frames, dtype=np.float32)

        min_lag = int(self.sample_rate / 400.0) # max 400 Hz
        max_lag = int(self.sample_rate / 70.0)  # min 70 Hz

        for i in range(n_frames):
            start = i * self.hop_len
            frame = audio[start : start + self.frame_len]
            if len(frame) < self.frame_len:
                break

            # Energy check
            energy = np.mean(frame ** 2)
            if energy < 1e-4:
                continue

            # Autocorrelation
            autocorr = np.correlate(frame, frame, mode='full')
            autocorr = autocorr[len(frame)-1:]

            if len(autocorr) > max_lag:
                search_region = autocorr[min_lag:max_lag]
                peak_idx = np.argmax(search_region) + min_lag
                r0 = autocorr[0]
                r_peak = autocorr[peak_idx]

                if r0 > 0 and (r_peak / r0) > 0.35:
                    f0 = self.sample_rate / peak_idx
                    f0_values[i] = f0
                    voiced_probs[i] = float(min(1.0, r_peak / r0))

        # Perform speaker log-F0 normalization
        norm_f0 = self.normalize_speaker_pitch(f0_values, voiced_probs)

        return PitchContour(
            time_stamps=time_stamps,
            f0_values=f0_values,
            voiced_probs=voiced_probs,
            normalized_f0=norm_f0
        )

    def normalize_speaker_pitch(self, f0_values: np.ndarray, voiced_probs: np.ndarray) -> np.ndarray:
        """
        Log-F0 Z-score speaker normalization.
        Z = (log(F0) - mean(log(F0))) / std(log(F0))
        """
        voiced_mask = voiced_probs > 0.5
        norm_f0 = np.zeros_like(f0_values)
        if np.sum(voiced_mask) > 1:
            log_f0 = np.log(f0_values[voiced_mask] + 1e-6)
            mean_log = np.mean(log_f0)
            std_log = np.std(log_f0) + 1e-6
            norm_f0[voiced_mask] = (np.log(f0_values[voiced_mask] + 1e-6) - mean_log) / std_log
        return norm_f0

    @staticmethod
    def compute_dtw_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
        """
        Dynamic Time Warping (DTW) distance metric for tone contour comparison.
        Returns similarity score in range [0, 1].
        """
        if len(seq1) == 0 or len(seq2) == 0:
            return 0.0

        n, m = len(seq1), len(seq2)
        dtw_matrix = np.full((n + 1, m + 1), fill_value=np.inf)
        dtw_matrix[0, 0] = 0.0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(seq1[i - 1] - seq2[j - 1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j],    # insertion
                    dtw_matrix[i, j - 1],    # deletion
                    dtw_matrix[i - 1, j - 1] # match
                )

        dist = dtw_matrix[n, m] / max(n, m)
        similarity = float(np.exp(-dist))
        return similarity
