import numpy as np
from src.evaluation.evaluator import CAPTEvaluator
from src.evaluation.benchmark import BenchmarkRunner, create_smoke_test_dataset
from src.application.assess_pronunciation import PronunciationAssessmentPipeline

def test_capt_evaluator_metrics():
    # Phonetic performance test
    targets = ["zh", "a", "m", "a"]
    preds = ["z", "a", "m", "a"] # 1 substitution
    gop = [0.4, 0.9, 0.95, 0.92]
    human = [1.0, 4.0, 4.5, 4.2]

    p_res = CAPTEvaluator.evaluate_phonetic_performance(targets, preds, gop, human)
    assert p_res["phoneme_accuracy"] == 0.75
    assert p_res["gop_human_correlation"] > 0.80

    # Tone performance test
    target_tones = [1, 2, 3, 4, 0]
    pred_tones = [1, 2, 3, 4, 1]
    t_res = CAPTEvaluator.evaluate_tone_performance(target_tones, pred_tones)
    assert t_res["tone_accuracy"] == 0.80
    assert t_res["macro_f1"] > 0.60

    # Diagnostic performance test
    det = [True, False, True, False]
    gt = [True, False, False, False]
    m_sev = [0.8, 0.1, 0.7, 0.2]
    h_sev = [0.9, 0.1, 0.6, 0.1]
    d_res = CAPTEvaluator.evaluate_diagnostics(det, gt, m_sev, h_sev)
    assert d_res["error_detection_precision"] == 0.5
    assert d_res["human_severity_correlation"] > 0.80

def test_benchmark_runner_on_mocks():
    runner = BenchmarkRunner()
    metrics = runner.run_benchmark()

    assert metrics["dataset_size"] >= 20
    assert "phonetics" in metrics
    assert "tone" in metrics
    assert "diagnostics" in metrics
    assert isinstance(metrics["phonetics"]["gop_human_correlation"], float)

def test_robustness_low_snr_rejection():
    # Audio with heavy noise causing low SNR should be safely rejected
    sr = 16000
    noise_audio = np.random.randn(sr).astype(np.float32) * 0.9

    pipeline = PronunciationAssessmentPipeline()
    res = pipeline.assess(noise_audio, sr, ["mā"], [1])

    # Audio quality metrics should flag low SNR and reject
    assert res["audio_quality"].snr_db < 10.0 or not res["audio_quality"].is_acceptable
    assert len(res["errors"]) >= 1
    assert res["errors"][0].category.value == "audio_quality" or res["errors"][0].category.value == "insufficient_evidence"
