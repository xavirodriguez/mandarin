from typing import List, Dict, Any
from src.domain.diagnosis.models import DiagnosticResult
from src.domain.diagnosis.feedback import PedagogicalFeedbackGenerator, PedagogicalFeedback

class FeedbackEngineService:
    """
    Application service that converts technical diagnostic results into structured pedagogical feedback.
    """

    def __init__(self):
        self.generator = PedagogicalFeedbackGenerator()

    def generate_feedback_reports(self, diagnostics: List[DiagnosticResult]) -> List[PedagogicalFeedback]:
        feedback_list = []
        for diag in diagnostics:
            fb = self.generator.generate_feedback(diag)
            feedback_list.append(fb)
        return feedback_list
