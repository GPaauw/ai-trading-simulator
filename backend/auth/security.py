import os
from fastapi import Header, HTTPException, status
from typing import Optional

from . import session

# Inloggegevens: vereis dat deze via omgevingsvariabelen worden gezet.
# Verwijder harde waarden uit de broncode voordat je de repo publiek maakt.
USERNAME = os.getenv("APP_USERNAME")
PASSWORD = os.getenv("APP_PASSWORD")


async def verify_token(authorization: Optional[str] = Header(None, alias="Authorization")) -> None:
    """Controleert of de Authorization header een geldige Bearer token bevat."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Ongeldige of ontbrekende token")
    token = authorization.split(" ", 1)[1]
    if not session.verify_token(token):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Ongeldige of verlopen token")


def validate_credentials(username: str, password: str) -> bool:
    if not USERNAME or not PASSWORD:
        # Credentials niet geconfigureerd — geen enkele login is toegestaan.
        return False
    return username == USERNAME and password == PASSWORD
