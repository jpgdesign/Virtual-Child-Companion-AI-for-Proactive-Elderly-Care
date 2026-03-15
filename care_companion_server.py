from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import uuid
import webbrowser
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

from llm_runtime import DEFAULT_ANALYSIS_PRESET, DEFAULT_GENERATION_PRESET
from platform_state import (
    authenticate_user,
    delete_conversation_record,
    delete_user,
    get_persona_profile,
    import_persona_profiles,
    list_login_accounts,
    load_platform_state,
    save_platform_state,
    sanitize_user,
    update_api_settings,
    update_persona_profile,
    update_prompt_settings,
    upsert_conversation_record,
    upsert_user,
)
from virtual_child_rl_system import VirtualChildRLSystem, normalize_algorithm


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "care_frontend"

AUTH_TOKENS: dict[str, Dict[str, Any]] = {}
SESSIONS: dict[str, Dict[str, Any]] = {}


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def issue_auth_token(user: Dict[str, Any]) -> str:
    token = uuid.uuid4().hex
    AUTH_TOKENS[token] = {
        **sanitize_user(user),
        "issued_at": _now_iso(),
    }
    return token


def get_auth_user(token: str | None, *, require_admin: bool = False) -> Dict[str, Any] | None:
    if not token:
        return None
    user = AUTH_TOKENS.get(token)
    if not user:
        return None
    if require_admin and user.get("role") != "admin":
        return None
    return user


def build_bootstrap_payload() -> Dict[str, Any]:
    state = load_platform_state()
    admin_accounts = [item for item in list_login_accounts(state, include_admin=True, include_password=True) if item["role"] == "admin"]
    return {
        "demo_accounts": list_login_accounts(state, include_admin=False, include_password=True),
        "admin_account": admin_accounts[0] if admin_accounts else None,
        "personas": state.get("personas", {}),
        "api_settings": state.get("api_settings", {}),
        "prompt_settings": state.get("prompt_settings", {}),
        "updated_at": state.get("updated_at", ""),
    }


def create_system_for_user(
    user: Dict[str, Any],
    *,
    algorithm: str | None = None,
    analysis_preset: str | None = None,
    generation_preset: str | None = None,
    llm_enabled: bool = True,
) -> VirtualChildRLSystem:
    state = load_platform_state()
    api_settings = state.get("api_settings", {})
    persona_profile = get_persona_profile(state, user.get("persona_profile_id", ""))
    return VirtualChildRLSystem(
        algorithm=normalize_algorithm(algorithm or api_settings.get("default_algorithm", "dqn")),
        llm_enabled=llm_enabled,
        analysis_preset=str(analysis_preset or api_settings.get("default_analysis_preset", DEFAULT_ANALYSIS_PRESET)),
        generation_preset=str(generation_preset or api_settings.get("default_generation_preset", DEFAULT_GENERATION_PRESET)),
        persona_profile_id=persona_profile["id"],
        persona_profile_data=persona_profile,
        prompt_settings=state.get("prompt_settings", {}),
    )


def serialize_session(
    session_id: str,
    system: VirtualChildRLSystem,
    latest_assistant_message: str | None,
    user: Dict[str, Any],
) -> dict:
    payload = system.build_ui_payload(latest_assistant_message=latest_assistant_message)
    payload["session_id"] = session_id
    payload["user"] = sanitize_user(user)
    return payload


def persist_session_record(session_id: str, system: VirtualChildRLSystem, user: Dict[str, Any]) -> None:
    state = load_platform_state()
    record = {
        "session_id": session_id,
        "username": user.get("username", ""),
        "display_name": user.get("display_name", ""),
        "role": user.get("role", ""),
        "persona_profile_id": user.get("persona_profile_id", ""),
        "persona_label": system.persona_profile.get("label", user.get("persona_profile_id", "")),
        "latest_assistant_message": system.session.latest_assistant_message,
        "summary": system.build_summary_dict(),
        "turns": system.serialize_turns(),
        "updated_at": _now_iso(),
    }
    upsert_conversation_record(state, record)


def create_user_session(user: Dict[str, Any], payload: Dict[str, Any]) -> tuple[str, Dict[str, Any], str]:
    system = create_system_for_user(
        user,
        algorithm=payload.get("algorithm"),
        analysis_preset=payload.get("analysis_preset"),
        generation_preset=payload.get("generation_preset"),
        llm_enabled=bool(payload.get("llm_enabled", True)),
    )
    opening_message = system.start_session()
    session_id = uuid.uuid4().hex
    SESSIONS[session_id] = {
        "system": system,
        "user": user,
        "created_at": _now_iso(),
    }
    persist_session_record(session_id, system, user)
    return session_id, SESSIONS[session_id], opening_message


def list_family_records(user: Dict[str, Any]) -> list[Dict[str, Any]]:
    state = load_platform_state()
    persona_profile_id = user.get("persona_profile_id")
    records = []
    for item in state.get("conversation_records", []):
        if item.get("persona_profile_id") != persona_profile_id:
            continue
        records.append(item)
    records.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
    return records


class CareCompanionHandler(BaseHTTPRequestHandler):
    server_version = "CareCompanionHTTP/2.0"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self._send_json({"ok": True, "service": "care_companion_server"})
            return

        self._serve_static(parsed.path)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        payload = self._read_json()

        if parsed.path == "/api/bootstrap":
            self._send_json(build_bootstrap_payload())
            return

        if parsed.path == "/api/login":
            state = load_platform_state()
            username = str(payload.get("username", "")).strip()
            password = str(payload.get("password", "")).strip()
            user = authenticate_user(state, username, password)
            if not user or user.get("role") == "admin":
                self._send_json({"error": "帳號或密碼錯誤。"}, status=HTTPStatus.UNAUTHORIZED)
                return
            token = issue_auth_token(user)
            self._send_json(
                {
                    "auth_token": token,
                    "user": user,
                    "bootstrap": build_bootstrap_payload(),
                }
            )
            return

        if parsed.path == "/api/admin/login":
            state = load_platform_state()
            username = str(payload.get("username", "")).strip()
            password = str(payload.get("password", "")).strip()
            admin_user = authenticate_user(state, username, password, require_admin=True)
            if not admin_user:
                self._send_json({"error": "後台帳號或密碼錯誤。"}, status=HTTPStatus.UNAUTHORIZED)
                return
            token = issue_auth_token(admin_user)
            self._send_json({"auth_token": token, "user": admin_user})
            return

        if parsed.path == "/api/logout":
            token = str(payload.get("auth_token", "")).strip()
            AUTH_TOKENS.pop(token, None)
            self._send_json({"ok": True})
            return

        if parsed.path == "/api/session":
            user = self._require_auth(payload)
            if not user:
                return
            session_id, session_bundle, opening_message = create_user_session(user, payload)
            self._send_json(serialize_session(session_id, session_bundle["system"], opening_message, user))
            return

        if parsed.path == "/api/reset":
            user = self._require_auth(payload)
            if not user:
                return
            session_id, session_bundle, opening_message = create_user_session(user, payload)
            self._send_json(serialize_session(session_id, session_bundle["system"], opening_message, user))
            return

        if parsed.path == "/api/chat":
            user = self._require_auth(payload)
            if not user:
                return
            session_id = str(payload.get("session_id", "")).strip()
            message = str(payload.get("message", "")).strip()
            session_bundle = SESSIONS.get(session_id)
            if not session_bundle or session_bundle.get("user", {}).get("username") != user.get("username"):
                self._send_json({"error": "Session not found. Please create a new conversation."}, status=HTTPStatus.NOT_FOUND)
                return
            if not message:
                self._send_json({"error": "Message is required."}, status=HTTPStatus.BAD_REQUEST)
                return

            system = session_bundle["system"]
            result = system.respond_fast(message)
            persist_session_record(session_id, system, user)
            self._send_json(serialize_session(session_id, system, result["assistant_message"], user))
            return

        if parsed.path == "/api/session_state":
            user = self._require_auth(payload)
            if not user:
                return
            session_id = str(payload.get("session_id", "")).strip()
            session_bundle = SESSIONS.get(session_id)
            if not session_bundle or session_bundle.get("user", {}).get("username") != user.get("username"):
                self._send_json({"error": "Session not found. Please create a new conversation."}, status=HTTPStatus.NOT_FOUND)
                return
            system = session_bundle["system"]
            persist_session_record(session_id, system, user)
            self._send_json(serialize_session(session_id, system, None, user))
            return

        if parsed.path == "/api/report":
            user = self._require_auth(payload)
            if not user:
                return
            records = list_family_records(user)
            latest_session_payload = None
            active_sessions = [
                (session_id, bundle)
                for session_id, bundle in SESSIONS.items()
                if bundle.get("user", {}).get("persona_profile_id") == user.get("persona_profile_id")
            ]
            if active_sessions:
                latest_session_id, latest_bundle = sorted(active_sessions, key=lambda item: item[1].get("created_at", ""), reverse=True)[0]
                latest_session_payload = serialize_session(latest_session_id, latest_bundle["system"], None, latest_bundle["user"])
            self._send_json({"records": records, "latest_session": latest_session_payload})
            return

        if parsed.path == "/api/admin/state":
            admin_user = self._require_auth(payload, require_admin=True)
            if not admin_user:
                return
            state = load_platform_state()
            self._send_json(
                {
                    "users": list_login_accounts(state, include_admin=True, include_password=True),
                    "personas": state.get("personas", {}),
                    "prompt_settings": state.get("prompt_settings", {}),
                    "api_settings": state.get("api_settings", {}),
                    "conversation_records": state.get("conversation_records", []),
                    "updated_at": state.get("updated_at", ""),
                }
            )
            return

        if parsed.path == "/api/admin/persona/update":
            admin_user = self._require_auth(payload, require_admin=True)
            if not admin_user:
                return
            state = load_platform_state()
            profile_id = str(payload.get("profile_id", "")).strip()
            profile = payload.get("profile")
            if not profile_id or not isinstance(profile, dict):
                self._send_json({"error": "profile_id and profile are required."}, status=HTTPStatus.BAD_REQUEST)
                return
            update_persona_profile(state, profile_id, profile)
            self._send_json({"ok": True})
            return

        if parsed.path == "/api/admin/persona/import":
            admin_user = self._require_auth(payload, require_admin=True)
            if not admin_user:
                return
            raw_text = str(payload.get("raw_text", "")).strip()
            if not raw_text:
                self._send_json({"error": "raw_text is required."}, status=HTTPStatus.BAD_REQUEST)
                return
            try:
                imported = json.loads(raw_text)
            except json.JSONDecodeError as exc:
                self._send_json({"error": f"JSON 解析失敗：{exc}"}, status=HTTPStatus.BAD_REQUEST)
                return
            state = load_platform_state()
            import_persona_profiles(state, imported)
            self._send_json({"ok": True})
            return

        if parsed.path == "/api/admin/prompts/update":
            admin_user = self._require_auth(payload, require_admin=True)
            if not admin_user:
                return
            state = load_platform_state()
            update_prompt_settings(state, payload.get("prompt_settings", {}))
            self._send_json({"ok": True})
            return

        if parsed.path == "/api/admin/users/upsert":
            admin_user = self._require_auth(payload, require_admin=True)
            if not admin_user:
                return
            state = load_platform_state()
            upsert_user(state, payload.get("user", {}))
            self._send_json({"ok": True})
            return

        if parsed.path == "/api/admin/users/delete":
            admin_user = self._require_auth(payload, require_admin=True)
            if not admin_user:
                return
            user_id = str(payload.get("user_id", "")).strip()
            if not user_id:
                self._send_json({"error": "user_id is required."}, status=HTTPStatus.BAD_REQUEST)
                return
            state = load_platform_state()
            delete_user(state, user_id)
            self._send_json({"ok": True})
            return

        if parsed.path == "/api/admin/api/update":
            admin_user = self._require_auth(payload, require_admin=True)
            if not admin_user:
                return
            state = load_platform_state()
            update_api_settings(state, payload.get("api_settings", {}))
            self._send_json({"ok": True})
            return

        if parsed.path == "/api/admin/records/delete":
            admin_user = self._require_auth(payload, require_admin=True)
            if not admin_user:
                return
            session_id = str(payload.get("session_id", "")).strip()
            if not session_id:
                self._send_json({"error": "session_id is required."}, status=HTTPStatus.BAD_REQUEST)
                return
            state = load_platform_state()
            delete_conversation_record(state, session_id)
            self._send_json({"ok": True})
            return

        self._send_json({"error": "Unknown endpoint."}, status=HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        logger.info("%s - %s", self.address_string(), format % args)

    def _read_json(self) -> dict:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            return {}
        raw_body = self.rfile.read(content_length)
        try:
            return json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _require_auth(self, payload: Dict[str, Any], *, require_admin: bool = False) -> Dict[str, Any] | None:
        token = str(payload.get("auth_token", "")).strip()
        user = get_auth_user(token, require_admin=require_admin)
        if not user:
            message = "需要先登入後台。" if require_admin else "需要先登入。"
            self._send_json({"error": message}, status=HTTPStatus.UNAUTHORIZED)
            return None
        return user

    def _serve_static(self, request_path: str) -> None:
        relative = request_path.lstrip("/") or "index.html"
        target = (FRONTEND_DIR / relative).resolve()
        if not str(target).startswith(str(FRONTEND_DIR.resolve())) or not target.exists() or not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return

        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        body = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8" if content_type.startswith("text/") else content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the interactive elderly-care frontend.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    load_platform_state()
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), CareCompanionHandler)
    url = f"http://{args.host}:{args.port}"
    logger.info("Care companion server running at %s", url)
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down care companion server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
