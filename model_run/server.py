"""Loopback-only HTTP host and a shared JSON dispatcher for ComfyUI."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import secrets
import socket
from urllib.parse import urlsplit

from .application import Application
from .trace_store import MAX_TRACE_BYTES, _unique_pairs

STATIC = Path(__file__).parent / "web"


class LocalHTTPServer(ThreadingHTTPServer):
    # Windows SO_REUSEADDR permits duplicate listeners with indeterminate routing.
    allow_reuse_address = False

    def server_bind(self):
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def decode_body(data):
    if len(data) > MAX_TRACE_BYTES:
        raise ValueError("Request exceeds 16 MiB")
    result = json.loads(data, object_pairs_hook=_unique_pairs,
                        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Invalid JSON number")))
    if not isinstance(result, dict):
        raise ValueError("Request must be a JSON object")
    return result


def dispatch(app, method, route, body=None):
    body = body or {}
    if method == "GET":
        if route == "state": return app.snapshot()
        if route == "catalog": return app.catalog()
        if route == "structure": return app.structure()
        if route.startswith("artifact-structure/"): return app.artifact_structure(route[19:])
        if route.startswith("inspect/"): return app.inspect(route[8:])
        if route.startswith("traces/"): return app.trace(route[7:])
        if route.startswith("native-workflow/"): return app.native_workflow(route[16:])
    if method == "POST":
        if route == "scan": return app.scan()
        if route == "load": return app.load(body.get("id"), body.get("device", "cpu"))
        if route == "unload": return app.unload()
        if route == "run": return app.run(body)
        if route == "cancel": return app.cancel()
        if route == "import": return app.publish(body, "Imported trace")
        if route.startswith("delete/"): return app.delete_trace(route[7:])
    raise LookupError("Unknown API endpoint")


def make_server(app, port=8766):
    token = secrets.token_urlsafe(32)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # Never log prompt or artifact payloads.

        def respond(self, status, data, content_type="application/json"):
            if not isinstance(data, bytes):
                data = json.dumps(data, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; object-src 'none'; frame-ancestors 'self'")
            self.end_headers()
            self.wfile.write(data)

        def handle_request(self):
            authority = f"127.0.0.1:{self.server.server_port}"
            # Exact Host + Origin checks prevent DNS rebinding and cross-site requests.
            if self.headers.get("Host") != authority:
                return self.respond(403, {"error": "Use the printed 127.0.0.1 URL"})
            origin = self.headers.get("Origin")
            if origin and origin != f"http://{authority}":
                return self.respond(403, {"error": "Cross-origin requests are disabled"})
            if self.headers.get("Sec-Fetch-Site") == "cross-site":
                return self.respond(403, {"error": "Cross-site requests are disabled"})
            route = urlsplit(self.path).path
            try:
                if route == "/api/session" and self.command == "GET":
                    return self.respond(200, {"token": token})
                if route.startswith("/api/"):
                    if self.headers.get("X-Model-Run-Token") != token:
                        return self.respond(403, {"error": "Missing application session token"})
                    body = None
                    if self.command == "POST":
                        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                            return self.respond(415, {"error": "Expected application/json"})
                        size = int(self.headers.get("Content-Length", "0"))
                        if not 0 < size <= MAX_TRACE_BYTES:
                            raise ValueError("Invalid request size")
                        self.connection.settimeout(15)
                        body = decode_body(self.rfile.read(size))
                    return self.respond(200, dispatch(app, self.command, route[5:], body))
                if self.command != "GET":
                    raise LookupError("Unknown endpoint")
                relative = "index.html" if route == "/" else route.lstrip("/")
                path = (STATIC / relative).resolve()
                if not path.is_relative_to(STATIC.resolve()) or not path.is_file():
                    raise LookupError("Asset not found; build the frontend first")
                self.respond(200, path.read_bytes(), mimetypes.guess_type(path)[0] or "application/octet-stream")
            except LookupError as exc:
                self.respond(404, {"error": str(exc)})
            except (ValueError, TypeError, OSError) as exc:
                self.respond(400, {"error": str(exc)[:2048]})
            except RuntimeError as exc:
                self.respond(409, {"error": str(exc)[:2048]})
            except Exception as exc:
                self.respond(500, {"error": f"{type(exc).__name__}: {exc}"[:2048]})

        do_GET = handle_request
        do_POST = handle_request
    return LocalHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description="Model Run local observatory")
    parser.add_argument("--models-dir", help="ComfyUI models root")
    parser.add_argument("--extra-paths", action="append", default=[])
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    from .comfy_paths import standard_model_paths, read_extra_paths
    entries = []
    if args.models_dir:
        entries.extend(standard_model_paths(args.models_dir)["entries"])
    for config in args.extra_paths:
        entries.extend(read_extra_paths(config)["entries"])
    entries = [entry for entry in entries if entry.get("exists", True)]
    app = Application(entries)
    server = make_server(app, args.port)
    print(f"Model Run: http://127.0.0.1:{server.server_port}", flush=True)
    app.scan()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.close()


if __name__ == "__main__":
    main()
