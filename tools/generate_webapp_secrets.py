#!/usr/bin/env python3
"""Generate WEBAPP_LINK_KEY and a strong WEBAPP_PASSWORD suggestion for Render."""

import secrets
import string

alphabet = string.ascii_letters + string.digits


def strong_password(length: int = 20) -> str:
    while True:
        pwd = "".join(secrets.choice(alphabet) for _ in range(length))
        if any(c.isalpha() for c in pwd) and any(c.isdigit() for c in pwd):
            return pwd


def main() -> None:
    link = secrets.token_urlsafe(32)
    pwd = strong_password()
    print("Add to Render → Environment (do not commit to git):\n")
    print(f"WEBAPP_LINK_KEY={link}")
    print(f"WEBAPP_PASSWORD={pwd}\n")
    print("Share this URL with authorised people only:")
    print(f"https://term-plan-mt.onrender.com/?k={link}")


if __name__ == "__main__":
    main()
