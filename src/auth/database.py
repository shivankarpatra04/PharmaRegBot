"""Supabase-backed user store with PBKDF2 password hashing.

User accounts live in a Postgres ``users`` table in your Supabase project
(accessed via the supabase-py client / PostgREST). Passwords are never stored in
plaintext — we keep our own PBKDF2-HMAC-SHA256 hashing with a per-user salt, and
only store the resulting hash/salt/iterations.

Create the table once by running ``supabase_schema.sql`` in the Supabase SQL
Editor. Use the **service_role** key (server-side) so the app can manage the
table while Row Level Security keeps it private from anon clients.

The public API (init_db, create_user, verify_user, get_user, user_exists,
list_users) is identical to the previous SQLite module, so the rest of the app
is unchanged.
"""

from __future__ import annotations

import hashlib
import re
import secrets
from datetime import datetime, timezone

from src.utils.config import Config, get_logger

logger = get_logger(__name__)

# PBKDF2 work factor. Stored per-user so it can be raised over time without
# breaking existing accounts.
_PBKDF2_ITERATIONS = 200_000
_MIN_PASSWORD_LEN = 6
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# Columns returned to the app (never includes password_hash / salt).
_PUBLIC_COLUMNS = "id, username, full_name, email, created_at"

_client = None


def _get_client():
    """Return a cached Supabase client, validating configuration lazily."""
    global _client
    if _client is None:
        if not Config.SUPABASE_URL or not Config.SUPABASE_KEY:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_KEY must be set in your .env file. "
                "Use the project URL and the service_role key from "
                "Supabase → Project Settings → API."
            )
        from supabase import create_client

        _client = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
    return _client


def _table():
    return _get_client().table(Config.SUPABASE_USERS_TABLE)


def init_db() -> None:
    """Verify the Supabase users table is reachable; guide the user if not."""
    try:
        _table().select("id").limit(1).execute()
    except ValueError:
        raise  # missing config — message is already actionable
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Could not access the Supabase users table "
            f"('{Config.SUPABASE_USERS_TABLE}'). Check SUPABASE_URL / SUPABASE_KEY "
            "and make sure you've run supabase_schema.sql in the SQL Editor. "
            f"Details: {exc}"
        ) from exc


def _hash_password(password: str, salt: str, iterations: int) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), iterations
    ).hex()


def _row_to_user(row: dict | None) -> dict | None:
    if not row:
        return None
    return {
        "id": row["id"],
        "username": row["username"],
        "full_name": row.get("full_name") or "",
        "email": row.get("email") or "",
        "created_at": row.get("created_at") or "",
    }


def _is_unique_violation(exc: Exception) -> bool:
    """Detect a Postgres unique-constraint violation across client error shapes."""
    if getattr(exc, "code", "") == "23505":
        return True
    text = str(getattr(exc, "message", "") or exc).lower()
    return (
        "duplicate key" in text
        or "already exists" in text
        or "unique constraint" in text
    )


def create_user(
    username: str, password: str, full_name: str = "", email: str = ""
) -> None:
    """Create a new account. Raises ValueError on validation/uniqueness errors."""
    username = (username or "").strip().lower()
    email = (email or "").strip().lower()

    if not username:
        raise ValueError("Username is required.")
    if " " in username:
        raise ValueError("Username cannot contain spaces.")
    if not email:
        raise ValueError("Email is required.")
    if not _EMAIL_RE.match(email):
        raise ValueError("Please enter a valid email address.")
    if not password or len(password) < _MIN_PASSWORD_LEN:
        raise ValueError(
            f"Password must be at least {_MIN_PASSWORD_LEN} characters long."
        )

    salt = secrets.token_hex(16)
    record = {
        "username": username,
        "full_name": (full_name or "").strip(),
        "email": email,
        "password_hash": _hash_password(password, salt, _PBKDF2_ITERATIONS),
        "salt": salt,
        "iterations": _PBKDF2_ITERATIONS,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        _table().insert(record).execute()
    except ValueError:
        raise  # missing config
    except Exception as exc:  # noqa: BLE001
        if _is_unique_violation(exc):
            raise ValueError("That username is already taken.")
        raise

    logger.info("Created user account '%s'", username)


def verify_user(username: str, password: str) -> dict | None:
    """Return a public user dict if credentials are valid, else None."""
    username = (username or "").strip().lower()
    if not username or not password:
        return None

    response = (
        _table().select("*").eq("username", username).limit(1).execute()
    )
    rows = response.data or []
    if not rows:
        return None

    row = rows[0]
    candidate = _hash_password(password, row["salt"], row["iterations"])
    # Constant-time comparison to avoid timing attacks.
    if secrets.compare_digest(candidate, row["password_hash"]):
        return _row_to_user(row)
    return None


def get_user(username: str) -> dict | None:
    """Return a user's public details by username (no secrets)."""
    username = (username or "").strip().lower()
    response = (
        _table()
        .select(_PUBLIC_COLUMNS)
        .eq("username", username)
        .limit(1)
        .execute()
    )
    rows = response.data or []
    return _row_to_user(rows[0]) if rows else None


def user_exists(username: str) -> bool:
    username = (username or "").strip().lower()
    response = _table().select("id").eq("username", username).limit(1).execute()
    return bool(response.data)


def list_users() -> list[dict]:
    """Return public details of all users (no secrets), ordered by id."""
    response = _table().select(_PUBLIC_COLUMNS).order("id").execute()
    return [_row_to_user(row) for row in (response.data or [])]
