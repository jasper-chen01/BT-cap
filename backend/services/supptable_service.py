"""
Firestore service for retrieving supplemental table URLs.
"""
from typing import Optional

from backend.config import settings

try:
    from google.cloud import firestore
    from google.oauth2 import service_account
except Exception:  # pragma: no cover - optional dependency
    firestore = None
    service_account = None


class SupptableService:
    """Fetch supplemental table URLs from Firestore."""

    def __init__(self):
        if firestore is None:
            raise RuntimeError("google-cloud-firestore is not installed.")

        credentials = None
        if settings.GOOGLE_APPLICATION_CREDENTIALS and service_account is not None:
            credentials = service_account.Credentials.from_service_account_file(
                settings.GOOGLE_APPLICATION_CREDENTIALS
            )

        project_id = settings.FIRESTORE_PROJECT_ID or getattr(credentials, "project_id", None)
        database_id = settings.FIRESTORE_DATABASE_ID or None
        self.client = firestore.Client(
            project=project_id,
            credentials=credentials,
            database=database_id,
        )
        self.collection = self.client.collection(settings.FIRESTORE_SUPPTABLE_COLLECTION)

    def get_supptable_url(self, doc_id: str) -> Optional[str]:
        doc_ref = self.collection.document(doc_id)
        snapshot = doc_ref.get()
        if not snapshot.exists:
            return None
        data = snapshot.to_dict() or {}
        for key in ("url", "download_url", "downloadUrl", "file_url", "fileUrl"):
            if data.get(key):
                return data[key]
        return None



