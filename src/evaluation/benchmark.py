import numpy as np
from typing import List, Dict, Any, Optional
from src.infrastructure.persistence.dataset import CAPTDatasetSample
from src.evaluation.evaluator import CAPTEvaluator
from src.application.assess_pronunciation import PronunciationAssessmentPipeline

def create_smoke_test_dataset(num_samples: int = 20, seed: int = 42) -> List[CAPTDatasetSample]:
    """
    Generates a deterministic 'set de humo' (smoke test dataset) of synthetic Mandarin audio samples
    with ground-truth human annotations for phonetics, tones, and error diagnostics.
    Used for baseline benchmarking when external L2 datasets (e.g. iCALL) are unavailable.
    """
    np.random.seed(seed)
    samples: List[CAPTDatasetSample] = []

    pinyin_candidates = [
        (["mā", "mα"], [1, 1], ["m", "a", "m", "a"], False, 0.0),
        (["ní", "hǎo"], [3, 3], ["n", "i", "h", "ao"], False, 0.0),
        (["zhī", "dào"], [1, 4], ["zh", "i", "d", "ao"], True, 0.75), # mispronounced 'zh' as 'z'
        (["yī", "yàng"], [1, 4], ["y", "i", "y", "ang"], False, 0.0),
        (["bù", "shì"], [4, 4], ["b", "u", "sh", "i"], True, 0.85), # tone error
    ]

    sr = 16000
    for i in range(num_samples):
        py, tones, ph, is_err, err_sev = pinyin_candidates[i % len(pinyin_candidates)]
        spk_id = f"speaker_{i % 5}"
        utt_id = f"utt_{i:03d}"

        # Generate a synthetic audio signal with tone modulation
        duration = 0.5 * len(py)
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        # Fundamental frequency 200 Hz with small harmonic components
        signal = 0.4 * np.sin(2 * np.pi * 200 * t) + 0.1 * np.sin(2 * np.pi * 400 * t)

        # Ground truth human rating (1.0 to 5.0 scale)
        human_phonetic_rating = 2.0 if is_err else 4.8
        human_tone_rating = 2.0 if (is_err and err_sev > 0.8) else 4.5

        sample = CAPTDatasetSample(
            speaker_id=spk_id,
            utterance_id=utt_id,
            target_text="".join(py),
            pinyin=py,
            phoneme_sequence=ph,
            syllable_boundaries=[],
            lexical_tones=tones,
            contextual_tones=tones,
            observed_pronunciation=ph,
            error_type="phonetic_confusion" if is_err else None,
            error_severity=err_sev,
            audio_path=None
        )
        # Attach synthetic audio and human ratings directly to object for benchmark harness
        sample.audio_waveform = signal
        sample.sample_rate = sr
        sample.human_phonetic_rating = human_phonetic_rating
        sample.human_tone_rating = human_tone_rating
        sample.has_error = is_err

        samples.append(sample)

    return samples

class BenchmarkRunner:
    """
    Evaluation Benchmark Harness that executes assessment pipelines over datasets
    and measures performance metrics against human annotations.
    """

    def __init__(self, pipeline: Optional[PronunciationAssessmentPipeline] = None):
        self.pipeline = pipeline or PronunciationAssessmentPipeline()

    def run_benchmark(self, dataset: Optional[List[CAPTDatasetSample]] = None) -> Dict[str, Any]:
        if dataset is None:
            dataset = create_smoke_test_dataset()

        gop_scores: List[float] = []
        human_phonetic_ratings: List[float] = []
        target_phonemes: List[str] = []
        pred_phonemes: List[str] = []

        target_tones: List[int] = []
        pred_tones: List[int] = []

        detected_errors: List[bool] = []
        gt_errors: List[bool] = []
        model_severities: List[float] = []
        human_severities: List[float] = []

        for sample in dataset:
            audio = getattr(sample, "audio_waveform", np.zeros(16000, dtype=np.float32))
            sr = getattr(sample, "sample_rate", 16000)

            result = self.pipeline.assess(
                waveform=audio,
                sample_rate=sr,
                target_pinyin=sample.pinyin,
                lexical_tones=sample.lexical_tones
            )

            # Extract metrics per syllable
            for syl_idx, syl in enumerate(result["syllables"]):
                target_phonemes.extend(syl.phonetics.target_phonemes)
                pred_phonemes.extend(syl.phonetics.detected_phonemes)

                gop_scores.append(syl.phonetics.overall_score)
                human_phonetic_ratings.append(getattr(sample, "human_phonetic_rating", 4.0))

                target_tones.append(syl.tone.contextual_target)
                pred_tones.append(syl.tone.predicted_tone)

            # Diagnostics
            has_detected_error = len(result["errors"]) > 0
            detected_errors.append(has_detected_error)
            gt_errors.append(getattr(sample, "has_error", False))

            max_model_sev = max([e.severity for e in result["errors"]], default=0.0)
            model_severities.append(max_model_sev)
            human_severities.append(sample.error_severity)

        # Compute benchmark evaluations
        phonetic_metrics = CAPTEvaluator.evaluate_phonetic_performance(
            target_phonemes, pred_phonemes, gop_scores, human_phonetic_ratings
        )
        tone_metrics = CAPTEvaluator.evaluate_tone_performance(target_tones, pred_tones)
        diag_metrics = CAPTEvaluator.evaluate_diagnostics(
            detected_errors, gt_errors, model_severities, human_severities
        )

        return {
            "dataset_size": len(dataset),
            "phonetics": phonetic_metrics,
            "tone": tone_metrics,
            "diagnostics": diag_metrics
        }
