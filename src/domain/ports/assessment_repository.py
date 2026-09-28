from typing import Protocol, Dict, Any, Optional

class AssessmentRepository(Protocol):
    """
    Domain Port Protocol for Assessment Result Persistence.
    Provides storage abstraction for pronunciation assessment records across SQLite/PostgreSQL adapters.
    """

    def save_assessment(self, assessment_data: Dict[str, Any], user_id: Optional[str] = None) -> str:
        """
        Persists assessment execution results (scores, target text, diagnostics) for a user.
        Returns persistent unique assessment record ID.
        """
        ...

    def get_assessment_by_id(self, assessment_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a persisted assessment record by ID.
        """
        ...
