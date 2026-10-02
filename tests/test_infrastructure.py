import numpy as np
from src.infrastructure.audio.qa import AudioQAProcessor
from src.infrastructure.pitch.processor import PitchProcessor
from src.infrastructure.alignment.aligner import forced_aligner_mock
from src.infrastructure.models.adapters import SpeechEncoderAdapter, PhonemeRecognizerAdapter, ToneClassifierAdapter
from src.infrastructure.calibration.calibrator import ModelCalibrator
from src.infrastructure.persistence.assessment_repository import SQLAssessmentRepository
from src.infrastructure.persistence.dataset import CAPTDatasetSample, CAPTDatasetLoader

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


def test_sql_assessment_repository(tmp_path):
    db_file = str(tmp_path / "test_assessments.db")
    repo = SQLAssessmentRepository(db_url=f"sqlite:///{db_file}")

    assessment_data = {
        "utterance": "ní hǎo",
        "phonetic_assessment": {"overall_score": 0.88},
        "tone_assessment": {"overall_score": 0.92},
        "errors": [{"category": "phonetic", "error_type": "tone_error", "severity": 0.3}]
    }

    record_id = repo.save_assessment(assessment_data, user_id="user_123")
    assert record_id is not None
    assert len(record_id) > 0

    record = repo.get_assessment_by_id(record_id)
    assert record is not None
    assert record["user_id"] == "user_123"
    assert record["target_text"] == "ní hǎo"
    assert record["phonetic_score"] == 0.88
    assert record["tone_score"] == 0.92
    assert len(record["diagnostics"]) == 1


def test_capt_dataset_loader():
    samples = [
        CAPTDatasetSample(
            speaker_id=f"S{i % 4}",
            utterance_id=f"U{i}",
            target_text="test",
            pinyin=["te", "st"],
            phoneme_sequence=["t", "e", "s", "t"],
            syllable_boundaries=[],
            lexical_tones=[1, 1],
            contextual_tones=[1, 1],
            observed_pronunciation=["t", "e", "s", "t"]
        )
        for i in range(20)
    ]

    loader = CAPTDatasetLoader(samples)
    train, val, test = loader.get_speaker_independent_splits(0.7, 0.15, 0.15)

    assert len(train) + len(val) + len(test) == 20

    train_speakers = set(s.speaker_id for s in train)
    val_speakers = set(s.speaker_id for s in val)
    test_speakers = set(s.speaker_id for s in test)

    assert train_speakers.isdisjoint(val_speakers)
    assert train_speakers.isdisjoint(test_speakers)
    assert val_speakers.isdisjoint(test_speakers)
