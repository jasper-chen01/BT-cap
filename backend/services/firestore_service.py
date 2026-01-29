"""
Firestore service for user auth storage
"""
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict

from passlib.context import CryptContext

from backend.config import settings

try:
    from google.cloud import firestore
    from google.cloud.firestore_v1 import FieldFilter
    from google.oauth2 import service_account
except Exception:  # pragma: no cover - optional dependency
    firestore = None
    FieldFilter = None
    service_account = None


class FirestoreService:
    """Simple Firestore user store."""

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
        self.collection = self.client.collection(settings.FIRESTORE_COLLECTION)
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    def get_user_by_email(self, email: str) -> Tuple[Optional[str], Optional[Dict]]:
        if FieldFilter is not None:
            query = self.collection.where(filter=FieldFilter("email", "==", email))
        else:
            query = self.collection.where("email", "==", email)
        docs = query.limit(1).stream()
        for doc in docs:
            return doc.id, doc.to_dict()
        return None, None

    def create_user(self, name: str, email: str, password: str) -> str:
        password_hash = self.pwd_context.hash(password)
        doc_ref = self.collection.document()
        doc_ref.set(
            {
                "name": name,
                "email": email,
                "password_hash": password_hash,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        return doc_ref.id

    def verify_password(self, password: str, password_hash: str) -> bool:
        return self.pwd_context.verify(password, password_hash)

