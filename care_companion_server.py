from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import uuid
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from llm_runtime import DEFAULT_ANALYSIS_PRESET, DEFAULT_GENERATION_PRESET
from virtual_child_rl_system import VirtualChildRLSystem, normalize_algorithm


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "care_frontend"
SESSIONS: dict[str, VirtualChildRLSystem] = {}


def create_system(
    algorithm: str,
    *,
    llm_enabled: bool = True,
    analysis_preset: str = DEFAULT_ANALYSIS_PRESET,
    generation_preset: str = DEFAULT_GENERATION_PRESET,
) -> VirtualChildRLSystem:
    return VirtualChildRLSystem(
        algorithm=normalize_algorithm(algorithm),
        llm_enabled=llm_enabled,
        analysis_preset=analysis_preset,
        generation_preset=generation_preset,
    )


def create_session(
    algorithm: str,
    *,
    llm_enabled: bool = True,
    analysis_preset: str = DEFAULT_ANALYSIS_PRESET,
    generation_preset: str = DEFAULT_GENERATION_PRESET,
) -> tuple[str, VirtualChildRLSystem, str]:
    system = create_system(
        algorithm,
        llm_enabled=llm_enabled,
        analysis_preset=analysis_preset,
        generation_preset=generation_preset,
    )
    opening_message = system.start_session()
    session_id = uuid.uuid4().hex
    SESSIONS[session_id] = system
    return session_id, system, opening_message


def serialize_session(session_id: str, system: VirtualChildRLSystem, latest_assistant_message: str | None) -> dict:
    payload = system.build_ui_payload(latest_assistant_message=latest_assistant_message)
    payload["session_id"] = session_id
    return payload


class CareCompanionHandler(BaseHTTPRequestHandler):
    server_version = "CareCompanionHTTP/1.0"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self._send_json({"ok": True, "service": "care_companion_server"})
            return

        self._serve_static(parsed.path)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        payload = self._read_json()

        if parsed.path == "/api/session":
            algorithm = normalize_algorithm(payload.get("algorithm", "dqn"))
            session_id, system, opening_message = create_session(
                algorithm,
                llm_enabled=bool(payload.get("llm_enabled", True)),
                analysis_preset=str(payload.get("analysis_preset", DEFAULT_ANALYSIS_PRESET)),
                generation_preset=str(payload.get("generation_preset", DEFAULT_GENERATION_PRESET)),
            )
            self._send_json(serialize_session(session_id, system, opening_message))
            return

        if parsed.path == "/api/reset":
            algorithm = normalize_algorithm(payload.get("algorithm", "dqn"))
            session_id, system, opening_message = create_session(
                algorithm,
                llm_enabled=bool(payload.get("llm_enabled", True)),
                analysis_preset=str(payload.get("analysis_preset", DEFAULT_ANALYSIS_PRESET)),
                generation_preset=str(payload.get("generation_preset", DEFAULT_GENERATION_PRESET)),
            )
            self._send_json(serialize_session(session_id, system, opening_message))
            return

        if parsed.path == "/api/chat":
            session_id = payload.get("session_id")
            message = str(payload.get("message", "")).strip()
            if not session_id or session_id not in SESSIONS:
                self._send_json({"error": "Session not found. Please reset the conversation."}, status=HTTPStatus.NOT_FOUND)
                return
            if not message:
                self._send_json({"error": "Message is required."}, status=HTTPStatus.BAD_REQUEST)
                return

            system = SESSIONS[session_id]
            result = system.respond_fast(message)
            self._send_json(serialize_session(session_id, system, result["assistant_message"]))
            return

        if parsed.path == "/api/session_state":
            session_id = payload.get("session_id")
            if not session_id or session_id not in SESSIONS:
                self._send_json({"error": "Session not found. Please reset the conversation."}, status=HTTPStatus.NOT_FOUND)
                return
            system = SESSIONS[session_id]
            self._send_json(serialize_session(session_id, system, None))
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
