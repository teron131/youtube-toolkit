"""Serve the read-only viewer UI and JSON API on loopback."""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import threading
from typing import cast
from urllib.parse import parse_qs, unquote, urlsplit
import webbrowser

from .reader import DEFAULT_PAGE_SIZE, ReadOnlyLanceViewer, TableNotFoundError, ViewerQueryError

VIEWER_HOST = "127.0.0.1"
LOGGER = logging.getLogger(__name__)
UI_DIRECTORY = Path(__file__).with_name("ui")
ASSET_SPECS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/viewer.css": ("viewer.css", "text/css; charset=utf-8"),
    "/viewer.js": ("viewer.js", "text/javascript; charset=utf-8"),
}
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; base-uri 'none'; connect-src 'self'; form-action 'self'; frame-ancestors 'none'; img-src 'self' data:; style-src 'self'; script-src 'self'"
)


class LanceViewerHTTPServer(ThreadingHTTPServer):
    """Threaded loopback server carrying the viewer and its packaged browser assets."""

    daemon_threads = True

    def __init__(
        self,
        viewer: ReadOnlyLanceViewer,
        assets: dict[str, tuple[str, bytes]],
        port: int,
    ) -> None:
        self.viewer = viewer
        self.assets = assets
        super().__init__((VIEWER_HOST, port), LanceViewerRequestHandler)


class LanceViewerRequestHandler(BaseHTTPRequestHandler):
    """Serve the viewer shell, static assets, and read-only JSON endpoints."""

    server: LanceViewerHTTPServer

    def do_GET(self) -> None:
        """Route one browser request without exposing mutation methods."""
        request = urlsplit(self.path)
        try:
            if request.path in self.server.assets:
                self.send_asset(request.path)
                return
            if request.path == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
                return
            if request.path == "/api/overview":
                self.send_json(self.server.viewer.overview())
                return
            table_name, wants_rows = parse_table_path(request.path)
            if wants_rows:
                query = parse_qs(request.query, keep_blank_values=True)
                self.send_json(
                    self.server.viewer.table_rows(
                        table_name,
                        offset=query_int(query, "offset", default=0),
                        limit=query_int(query, "limit", default=DEFAULT_PAGE_SIZE),
                        sort_column=query_value(query, "sort"),
                        sort_direction=query_value(query, "direction") or "asc",
                        search_text=query_value(query, "search"),
                        filter_column=query_value(query, "filter_column"),
                        filter_operator=query_value(query, "filter_operator"),
                        filter_value=query_value(query, "filter_value"),
                    )
                )
                return
            self.send_json(self.server.viewer.table_detail(table_name))
        except TableNotFoundError as error:
            self.send_error_json(HTTPStatus.NOT_FOUND, str(error))
        except ViewerQueryError as error:
            self.send_error_json(HTTPStatus.BAD_REQUEST, str(error))
        except Exception as error:
            LOGGER.exception("LanceDB viewer request failed")
            self.send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))

    def do_POST(self) -> None:
        """Reject writes explicitly instead of exposing the base handler's generic response."""
        self.send_error_json(HTTPStatus.METHOD_NOT_ALLOWED, "viewer API is read-only")

    do_PUT = do_POST
    do_PATCH = do_POST
    do_DELETE = do_POST

    def send_asset(self, route: str) -> None:
        """Return one allowlisted packaged UI asset with local-only browser policies."""
        content_type, body = self.server.assets[route]
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", CONTENT_SECURITY_POLICY)
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(body)

    def send_json(
        self,
        payload: object,
        *,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        """Serialize one strict JSON response for the local viewer."""
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, status: HTTPStatus, message: str) -> None:
        """Keep API failures machine-readable for inline browser error states."""
        self.send_json({"error": message}, status=status)

    def log_message(self, format: str, *args: object) -> None:
        """Keep routine local browser requests out of command output."""
        del format, args


def parse_table_path(path: str) -> tuple[str, bool]:
    """Parse the two allowlisted table API route shapes."""
    prefix = "/api/tables/"
    if not path.startswith(prefix):
        raise TableNotFoundError(f"endpoint does not exist: {path}")
    remainder = path.removeprefix(prefix)
    wants_rows = remainder.endswith("/rows")
    encoded_name = remainder.removesuffix("/rows") if wants_rows else remainder
    if not encoded_name or "/" in encoded_name:
        raise TableNotFoundError(f"endpoint does not exist: {path}")
    return unquote(encoded_name), wants_rows


def query_value(query: dict[str, list[str]], name: str) -> str | None:
    """Return one optional query-string value and reject duplicates."""
    values = query.get(name)
    if values is None:
        return None
    if len(values) != 1:
        raise ViewerQueryError(f"query parameter may appear once: {name}")
    return values[0]


def query_int(
    query: dict[str, list[str]],
    name: str,
    *,
    default: int,
) -> int:
    """Parse one integer query-string value with a viewer-readable error."""
    raw_value = query_value(query, name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError as error:
        raise ViewerQueryError(f"query parameter must be an integer: {name}") from error


def create_database_viewer_server(
    database_path: Path,
    *,
    port: int = 0,
) -> LanceViewerHTTPServer:
    """Create the live viewer server without starting its blocking loop."""
    if not 0 <= port <= 65_535:
        raise ValueError("port must be between 0 and 65535")
    viewer = ReadOnlyLanceViewer(database_path)
    assets = {
        route: (
            content_type,
            (UI_DIRECTORY / filename).read_bytes(),
        )
        for route, (filename, content_type) in ASSET_SPECS.items()
    }
    try:
        return LanceViewerHTTPServer(viewer, assets, port)
    except OSError as error:
        raise RuntimeError(f"Could not bind LanceDB viewer to {VIEWER_HOST}:{port}: {error}") from error


def serve_database_viewer(
    database_path: Path,
    *,
    port: int = 0,
    open_browser: bool = True,
) -> None:
    """Run the loopback viewer until interrupted and optionally open its browser tab."""
    server = create_database_viewer_server(database_path, port=port)
    bound_host, bound_port = cast(tuple[str, int], server.server_address)
    url = f"http://{bound_host}:{bound_port}"
    print(f"LanceDB viewer  {url}")
    print(f"Database        {database_path.resolve()}")
    print("Read only       Press Ctrl-C to stop")
    if open_browser:
        threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
