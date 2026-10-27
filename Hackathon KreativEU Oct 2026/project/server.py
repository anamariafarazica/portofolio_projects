"""
Local server for the Luigi FM newsroom game. Uses only Python's standard library.

Run:  python server.py
Then open  http://127.0.0.1:8000            (the game)
      or   http://127.0.0.1:8000/dashboard  (the headline test desk)

Keep all project files in the same folder.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

print("Loading the language model for the AI editor (this takes a few seconds)...")
from headline_checker import review  # noqa: E402  (loads bart once, at startup)
import newsroom  # noqa: E402

HERE = Path(__file__).parent
PORT = 8000
PAGES = {"/": "game.html", "/game": "game.html", "/dashboard": "dashboard.html"}
# Requests are handled in parallel so the page never hangs,
# but the game state and the models are used one request at a time.
LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?")[0]
        if path in PAGES:
            return self._send(200, (HERE / PAGES[path]).read_bytes(), "text/html; charset=utf-8")
        if path == "/api/game":
            with LOCK:
                return self._json(200, newsroom.view())
        if path == "/api/status":
            ready, message = newsroom.ollama_status()
            return self._json(200, {"ollama": ready, "message": message, "api": newsroom.API,
                                    "folder": str(Path(newsroom.__file__).resolve().parent)})
        self._send(404, b"Not found", "text/plain")

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length) or b"{}")
            path = self.path.split("?")[0]
            key = str(data.get("key", ""))
            headline = str(data.get("headline", "")).strip()

            with LOCK:
                if path == "/api/new":
                    newsroom.new_game()
                    return self._json(200, newsroom.view())
                if path in ("/api/check", "/api/publish") and not headline:
                    return self._json(400, {"error": "Write a headline first."})
                if path == "/api/check":
                    return self._json(200, newsroom.check(key, headline))
                if path == "/api/publish":
                    result = newsroom.publish(key, headline, bool(data.get("unconfirmed")))
                    return self._json(200, {**result, "game": newsroom.view()})
                if path == "/api/judge":
                    newsroom.judge(key, str(data.get("label", "")))
                    return self._json(200, {"game": newsroom.view()})
                if path == "/api/verify":
                    newsroom.verify(key)
                    return self._json(200, {"game": newsroom.view()})
                if path == "/api/hold":
                    newsroom.hold(key)
                    return self._json(200, {"game": newsroom.view()})
                if path == "/api/end_cycle":
                    return self._json(200, newsroom.end_cycle())
                if path == "/api/next_story":
                    return self._json(200, newsroom.next_story())
                if path == "/api/open_case":
                    return self._json(200, {"game": newsroom.open_case(str(data.get("case", "")))})
                if path == "/api/career":
                    return self._json(200, newsroom.finish_career())
                if path == "/check":  # the headline test desk
                    notebook = [e for e in data.get("notebook", []) if str(e.get("text", "")).strip()]
                    if not headline:
                        return self._json(400, {"error": "Write a headline first."})
                    if not notebook:
                        return self._json(400, {"error": "Add at least one notebook entry."})
                    return self._json(200, review(headline, notebook))
            self._send(404, b"Not found", "text/plain")
        except ValueError as error:  # a game rule, e.g. deciding twice
            self._json(400, {"error": str(error)})
        except Exception as error:  # show the real problem in the page
            self._json(500, {"error": f"{type(error).__name__}: {error}"})

    def _json(self, code, payload):
        self._send(code, json.dumps(payload).encode(), "application/json")

    def _send(self, code, body, content_type):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # keep the terminal quiet
        pass


if __name__ == "__main__":
    print(f"Game files: {Path(newsroom.__file__).resolve().parent}  (engine version {newsroom.API})")
    ready, message = newsroom.ollama_status()
    print(("OK: " if ready else "Note: ") + message)
    if not ready:
        print("      The game still works, using the pre-written reader comments from the case files.")
    try:
        server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    except OSError:
        raise SystemExit(f"Port {PORT} is already in use. Close the other server (Ctrl+C in its terminal) and try again.")
    server.daemon_threads = True
    print(f"Game running at http://127.0.0.1:{PORT}  (press Ctrl+C to stop)")
    server.serve_forever()
