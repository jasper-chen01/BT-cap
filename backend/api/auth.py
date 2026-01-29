"""
Auth endpoints for signup/signin using Firestore
"""
from fastapi import APIRouter, HTTPException

from backend.models.schemas import SignUpRequest, SignInRequest, AuthResponse
from backend.services.firestore_service import FirestoreService

router = APIRouter()


@router.post("/auth/signup", response_model=AuthResponse)
async def signup(payload: SignUpRequest):
    """Create a new user account."""
    store = FirestoreService()
    _, existing = store.get_user_by_email(payload.email)
    if existing:
        raise HTTPException(status_code=409, detail="Account already exists")

    user_id = store.create_user(payload.name, payload.email, payload.password)
    return AuthResponse(user_id=user_id, name=payload.name, email=payload.email)


@router.post("/auth/signin", response_model=AuthResponse)
async def signin(payload: SignInRequest):
    """Sign in an existing user."""
    store = FirestoreService()
    user_id, user = store.get_user_by_email(payload.email)
    if not user or not user_id:
        raise HTTPException(status_code=404, detail="Account not found")

    if not store.verify_password(payload.password, user.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return AuthResponse(user_id=user_id, name=user.get("name", ""), email=user["email"])

