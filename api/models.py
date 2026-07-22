from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


Role = Literal["admin", "user", "viewer"]


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class RefreshRequest(BaseModel):
    refresh_token: str = Field(min_length=1)


class LogoutRequest(BaseModel):
    refresh_token: str | None = None
    all: bool = False


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=64, pattern=r"^[\w.-]+$")
    password: str = Field(min_length=10, max_length=256)
    role: Role


class UserUpdate(BaseModel):
    role: Role | None = None
    password: str | None = Field(default=None, min_length=10, max_length=256)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=8000)
    conversation_id: str | None = None
    top_k: int = Field(default=10, ge=1, le=50)
    threshold: float | None = Field(default=None, ge=0, le=1)


class ConversationCreate(BaseModel):
    title: str | None = Field(default=None, max_length=160)


class ProviderActivate(BaseModel):
    id: str = Field(min_length=1, max_length=64)


class ProviderModelUpdate(BaseModel):
    model: str = Field(min_length=1, max_length=200)


class ThresholdUpdate(BaseModel):
    threshold: float = Field(ge=0, le=1)
