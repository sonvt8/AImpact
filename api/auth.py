from __future__ import annotations

import hashlib
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from threading import Lock
from uuid import uuid4

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from api.db import Database
from api.settings import Settings


password_hasher = PasswordHasher()
bearer = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    return password_hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return password_hasher.verify(password_hash, password)
    except (VerifyMismatchError, InvalidHashError):
        return False


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _encode(user: dict, token_type: str, expires_at: datetime, settings: Settings) -> str:
    now = datetime.now(timezone.utc)
    return jwt.encode(
        {
            "sub": str(user["id"]),
            "username": user["username"],
            "role": user["role"],
            "type": token_type,
            "jti": str(uuid4()),
            "iat": now,
            "exp": expires_at,
        },
        settings.jwt_secret,
        algorithm="HS256",
    )


def issue_tokens(user: dict, db: Database, settings: Settings) -> dict:
    now = datetime.now(timezone.utc)
    access_expiry = now + timedelta(minutes=settings.jwt_access_expire_min)
    refresh_expiry = now + timedelta(days=settings.jwt_refresh_expire_days)
    access_token = _encode(user, "access", access_expiry, settings)
    refresh_token = _encode(user, "refresh", refresh_expiry, settings)
    db.store_refresh(token_hash(refresh_token), user["id"], refresh_expiry.isoformat())
    return {"access_token": access_token, "refresh_token": refresh_token}


def decode_token(token: str, settings: Settings, expected_type: str) -> dict:
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
    except JWTError as error:
        raise HTTPException(status_code=401, detail="Invalid or expired token") from error
    if payload.get("type") != expected_type or not payload.get("sub"):
        raise HTTPException(status_code=401, detail="Invalid token type")
    return payload


def rotate_refresh(refresh_token: str, db: Database, settings: Settings) -> tuple[dict, dict]:
    payload = decode_token(refresh_token, settings, "refresh")
    stored = db.active_refresh(token_hash(refresh_token))
    if not stored or stored["user_id"] != int(payload["sub"]):
        raise HTTPException(status_code=401, detail="Refresh token has been revoked")
    user = db.get_user(int(payload["sub"]))
    if not user:
        raise HTTPException(status_code=401, detail="User no longer exists")
    db.revoke_refresh(token_hash(refresh_token))
    return issue_tokens(user, db, settings), user


async def current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> dict:
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    payload = decode_token(credentials.credentials, request.app.state.settings, "access")
    user = request.app.state.db.get_user(int(payload["sub"]))
    if not user:
        raise HTTPException(status_code=401, detail="User no longer exists")
    return user


def require_roles(*roles: str):
    async def dependency(user: dict = Depends(current_user)) -> dict:
        if user["role"] not in roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role")
        return user

    return dependency


class LoginRateLimiter:
    # ponytail: per-process limiter; move to Redis only when running multiple API replicas.
    def __init__(self, limit: int = 5, window_seconds: int = 60):
        self.limit = limit
        self.window_seconds = window_seconds
        self.attempts: dict[str, deque[float]] = defaultdict(deque)
        self.lock = Lock()

    def allow(self, key: str, now: float) -> bool:
        with self.lock:
            attempts = self.attempts[key]
            while attempts and attempts[0] <= now - self.window_seconds:
                attempts.popleft()
            if len(attempts) >= self.limit:
                return False
            attempts.append(now)
            return True

    def clear(self, key: str) -> None:
        with self.lock:
            self.attempts.pop(key, None)
