import numpy as np
from src.infrastructure.audio.qa import AudioQAProcessor
from src.infrastructure.pitch.processor import PitchProcessor
from src.infrastructure.alignment.aligner import forced_aligner_mock
from src.infrastructure.models.adapters import SpeechEncoderAdapter, PhonemeRecognizerAdapter, ToneClassifierAdapter
from src.infrastructure.calibration.calibrator import ModelCalibrator

def test_audio_qa_processor():
    sr = 16000
    # Generate 1 sec sine wave
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    processor = AudioQAProcessor()
    proc_audio, metrics = processor.process_and_verify(audio, sr)

    assert len(proc_audio) == sr
    assert metrics.is_acceptable is True
    assert metrics.clipping_ratio == 0.0

def test_pitch_processor():
    sr = 16000
    t = np.linspace(0, 0.5, int(sr * 0.5), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 220 * t) # 220 Hz pitch

    pitch_proc = PitchProcessor()
    contour = pitch_proc.estimate_pitch(audio)

    assert len(contour.time_stamps) > 0
    assert contour.mean_f0 > 0.0

def test_aligner_and_adapters():
    audio = np.zeros(16000, dtype=np.float32)
    aligner = forced_aligner_mock()
    alignments = aligner.align(audio, 16000, ["mā", "fan"])

    assert len(alignments) == 2
    assert alignments[0]["pinyin"] == "mā"
    assert alignments[0]["onset"] is not None

    encoder = SpeechEncoderAdapter()
    feat = encoder.encode(audio, 16000)
    assert feat.shape[0] > 0

    recognizer = PhonemeRecognizerAdapter()
    post = recognizer.recognize_posteriors(feat)
    assert post.shape[0] == feat.shape[0]

def test_calibrator():
    calibrator = ModelCalibrator(temperature=1.5)
    probs = {1: 0.8, 2: 0.1, 3: 0.05, 4: 0.05, 0: 0.0}
    calibrated = calibrator.calibrate_probabilities(probs)

    assert abs(sum(calibrated.values()) - 1.0) < 1e-5
    assert calibrated[1] < 0.8 # smoothed by temperature > 1
