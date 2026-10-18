"""Server-side identity and workspace derivation.

The workspace is always derived from the authenticated principal; any
client-supplied ``workspace_id`` is ignored.
"""

from __future__ import annotations

import logging
import re
import secrets
from dataclasses import dataclass
from functools import lru_cache

from .config import Settings

logger = logging.getLogger(__name__)

LEGACY_WORKSPACE = "default"
_LOCAL_ENVIRONMENTS = {"local", "dev", "development", "test", "demo"}


class AuthError(Exception):
    """Raised when credentials are missing or invalid."""


@dataclass(frozen=True)
class Principal:
    user_id: str
    workspace_id: str
    method: str  # "jwt" | "basic" | "anonymous"


def jwt_configured(settings: Settings) -> bool:
    # Explicit opt-in: SUPABASE_URL alone (used for pgvector) must not switch on JWT-only auth.
    return bool(settings.supabase_jwt_secret or settings.supabase_jwks_url)


def basic_configured(settings: Settings) -> bool:
    return bool(settings.basic_auth_username or settings.basic_auth_password)


def auth_configured(settings: Settings) -> bool:
    return jwt_configured(settings) or basic_configured(settings)


def validate_auth_config(settings: Settings) -> None:
    """Fail closed: refuse to run a non-local deployment without any auth."""
    if auth_configured(settings):
        return
    if settings.environment.lower() in _LOCAL_ENVIRONMENTS:
        logger.warning("auth.disabled environment=%s -- API is unauthenticated", settings.environment)
        return
    raise RuntimeError(
        "No authentication configured. Set CAREMIND_USERNAME/CAREMIND_PASSWORD or Supabase JWT "
        "settings, or run with CAREMIND_ENV=local."
    )


def workspace_for_user(user_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]", "-", user_id)[:80]
    return f"user-{safe}"


@lru_cache(maxsize=4)
def _jwks_client(url: str):
    import jwt

    return jwt.PyJWKClient(url, cache_keys=True)


def verify_jwt(token: str, settings: Settings) -> str:
    """Verify a Supabase access token and return its subject (user id)."""
    import jwt

    audience = settings.supabase_jwt_audience or None
    options = {"require": ["exp", "sub"], "verify_aud": bool(audience)}
    try:
        header = jwt.get_unverified_header(token)
        alg = header.get("alg", "")
        if alg == "HS256":
            if not settings.supabase_jwt_secret:
                raise AuthError("HS256 tokens are not accepted")
            claims = jwt.decode(
                token, settings.supabase_jwt_secret, algorithms=["HS256"], audience=audience, options=options
            )
        elif alg in {"RS256", "ES256"}:
            jwks_url = settings.supabase_jwks_url
            if not jwks_url:
                raise AuthError("Asymmetric tokens are not accepted")
            key = _jwks_client(jwks_url).get_signing_key_from_jwt(token).key
            claims = jwt.decode(token, key, algorithms=[alg], audience=audience, options=options)
        else:
            raise AuthError("Unsupported token algorithm")
    except AuthError:
        raise
    except Exception as exc:  # jwt.PyJWTError, network errors fetching JWKS
        logger.info("auth.jwt_invalid error=%s", type(exc).__name__)
        raise AuthError("Invalid or expired token") from exc
    sub = str(claims.get("sub") or "")
    if not sub:
        raise AuthError("Token has no subject")
    return sub


def authenticate(settings: Settings, *, bearer: str | None, basic: tuple[str, str] | None) -> Principal:
    if not auth_configured(settings):
        return Principal(user_id="anonymous", workspace_id=LEGACY_WORKSPACE, method="anonymous")

    if bearer:
        if not jwt_configured(settings):
            raise AuthError("Bearer tokens are not accepted")
        user_id = verify_jwt(bearer, settings)
        return Principal(user_id=user_id, workspace_id=workspace_for_user(user_id), method="jwt")

    if basic is not None and basic_configured(settings):
        user_ok = secrets.compare_digest(basic[0], settings.basic_auth_username or "")
        pass_ok = secrets.compare_digest(basic[1], settings.basic_auth_password or "")
        if user_ok and pass_ok:
            # The single shared login keeps the legacy workspace so existing data stays reachable.
            return Principal(user_id=basic[0], workspace_id=LEGACY_WORKSPACE, method="basic")
        raise AuthError("Invalid credentials")

    raise AuthError("Authentication required")
