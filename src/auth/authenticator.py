"""Streamlit authentication flow.

Renders login / register forms when the user is not authenticated and exposes
helpers for reading the current user and logging out. Login state is held in
``st.session_state`` for the duration of the browser session.
"""

from __future__ import annotations

import streamlit as st

from src.auth import database
from src.utils.config import get_logger

logger = get_logger(__name__)


@st.cache_resource(show_spinner=False)
def ensure_db_initialised() -> bool:
    """Create the users table once per process."""
    database.init_db()
    return True


def current_user() -> dict | None:
    """Return the logged-in user dict, or None."""
    return st.session_state.get("user")


def logout() -> None:
    """Clear the session: user + any conversation state."""
    st.session_state.pop("user", None)
    st.session_state["messages"] = []
    st.session_state["total_tokens"] = 0


def _render_login_tab() -> None:
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", use_container_width=True, type="primary")

    if submitted:
        user = database.verify_user(username, password)
        if user:
            st.session_state["user"] = user
            # Start each login with a clean conversation.
            st.session_state["messages"] = []
            st.session_state["total_tokens"] = 0
            logger.info("User '%s' logged in", user["username"])
            st.rerun()
        else:
            st.error("Invalid username or password.")


def _render_register_tab() -> None:
    with st.form("register_form", clear_on_submit=False):
        full_name = st.text_input("Full name (optional)")
        username = st.text_input("Choose a username", help="Lowercase, no spaces.")
        email = st.text_input("Email", help="e.g. name@company.com")
        password = st.text_input("Choose a password", type="password", help="Min. 6 characters.")
        confirm = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Create account", use_container_width=True)

    if submitted:
        if password != confirm:
            st.error("Passwords do not match.")
            return
        try:
            database.create_user(username, password, full_name, email)
            st.success("✅ Account created! Switch to the **Login** tab to sign in.")
        except ValueError as exc:
            st.error(str(exc))


def render_auth_ui() -> None:
    """Render the full-page authentication screen (login + register)."""
    ensure_db_initialised()

    # Centre the auth card.
    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.markdown("# PharmaRegBot 🧪")
        st.caption("Regulatory Document Intelligence System")
        st.markdown(
            "Sign in to upload your regulatory documents and ask grounded, "
            "cited questions. Your documents are private to your account."
        )
        login_tab, register_tab = st.tabs(["🔐 Login", "📝 Register"])
        with login_tab:
            _render_login_tab()
        with register_tab:
            _render_register_tab()
