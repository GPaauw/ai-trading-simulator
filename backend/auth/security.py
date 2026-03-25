import os
from fastapi import Header, HTTPException, status
from typing import Optional

from . import session

# Inloggegevens (houd deze buiten git in productie door env vars te gebruiken)
USERNAME = os.getenv("APP_USERNAME", "admin")
PASSWORD = os.getenv("APP_PASSWORD", "trading123")


async def verify_token(authorization: Optional[str] = Header(None, alias="Authorization")) -> None:
    """Controleert of de Authorization header een geldige Bearer token bevat."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Ongeldige of ontbrekende token")
    token = authorization.split(" ", 1)[1]
    if not session.verify_token(token):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Ongeldige of verlopen token")


def validate_credentials(username: str, password: str) -> bool:
    return username == USERNAME and password == PASSWORD
