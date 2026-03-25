import base64
import hashlib
import hmac
import os
import time

_TTL_SECONDS = 4 * 60 * 60


def _secret_key() -> bytes:
    secret = os.getenv("TOKEN_SECRET") or os.getenv("APP_PASSWORD") or "dev-token-secret"
    return secret.encode("utf-8")


def _sign(payload: str) -> str:
    digest = hmac.new(_secret_key(), payload.encode("utf-8"), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")


def create_token() -> str:
    expires_at = int(time.time()) + _TTL_SECONDS
    payload = str(expires_at)
    signature = _sign(payload)
    return f"{payload}.{signature}"


def verify_token(token: str) -> bool:
    if not token or "." not in token:
        return False

    payload, signature = token.rsplit(".", 1)
    if not hmac.compare_digest(signature, _sign(payload)):
        return False

    try:
        expires_at = int(payload)
    except ValueError:
        return False

    return int(time.time()) <= expires_at


def invalidate_token(token: str) -> None:
    # Stateless tokens kunnen niet server-side ongeldig gemaakt worden zonder
    # extra opslag. Uitloggen gebeurt daarom client-side door de token te wissen.
    return None
