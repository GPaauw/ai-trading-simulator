import secrets
from datetime import datetime, timedelta
from typing import Dict

# Eenvoudige in-memory sessies (niet persistent). Geschikt voor dev/demo.
_TOKENS: Dict[str, datetime] = {}
_TTL = timedelta(hours=4)


def create_token() -> str:
    token = secrets.token_urlsafe(32)
    _TOKENS[token] = datetime.utcnow() + _TTL
    return token


def verify_token(token: str) -> bool:
    if not token:
        return False
    exp = _TOKENS.get(token)
    if not exp:
        return False
    if datetime.utcnow() > exp:
        # token verlopen
        del _TOKENS[token]
        return False
    return True


def invalidate_token(token: str) -> None:
    _TOKENS.pop(token, None)
