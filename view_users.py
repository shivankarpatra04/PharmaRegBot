"""View all registered PharmaRegBot users (id, username, full name, email, joined).

Passwords are stored only as salted PBKDF2 hashes and are never shown.

Usage:
    python view_users.py
"""

from __future__ import annotations

from src.auth.database import init_db, list_users
from src.utils.config import Config


def main() -> None:
    init_db()  # verify the Supabase users table is reachable
    users = list_users()

    print(f"Supabase project: {Config.SUPABASE_URL}")
    print(f"Users table: {Config.SUPABASE_USERS_TABLE}")
    print(f"Total users: {len(users)}\n")

    if not users:
        print("No users registered yet.")
        return

    headers = ("ID", "USERNAME", "FULL NAME", "EMAIL", "JOINED (UTC)")
    rows = [
        (
            str(u["id"]),
            u["username"],
            u["full_name"] or "-",
            u["email"] or "-",
            (u["created_at"] or "")[:19].replace("T", " "),
        )
        for u in users
    ]

    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))
    ]
    fmt = "  ".join("{:<" + str(w) + "}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*("-" * w for w in widths)))
    for row in rows:
        print(fmt.format(*row))


if __name__ == "__main__":
    main()
