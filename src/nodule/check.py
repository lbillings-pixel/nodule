"""The owner's check page for the proposed labels (ADR 0004, choice 9b).

`python -m nodule.check` serves a page on this PC only (127.0.0.1) and opens it
in the browser. It shows each photo of the fixed check sample
(`labels.check_sample`) as its outline-check picture, with Claude's proposed
labels filled in; the owner confirms or corrects them, may add a rough band
count by eye, and saves. Each save is checked (`labels.problems`) and written to
`references/library-view-labels.toml` as the photo's `checked` table. The page
does not show the measured band count, so the eye count is not led by it.

Pictures come from `runs/outline-check/` (`python -m nodule.overlay`): Commons
photos only, never the owner's specimens. Nothing leaves the PC.
"""

from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from collections.abc import Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from nodule import labels
from nodule.overlay import OUT_DIR

PAGE = Path(__file__).with_name("check.html")
HOST = "127.0.0.1"
PORT = 8765


class Checker:
    """The sample and the labels file, with saves taken one at a time."""

    def __init__(self, path: Path = labels.LABELS_TOML, pictures: Path = OUT_DIR) -> None:
        self.path = path
        self.pictures = pictures
        self.images = labels.load(path)
        self.sample = [k for k in labels.check_sample(self.images) if "proposed" in self.images[k]]
        self.lock = threading.Lock()

    def rows(self) -> list[dict[str, object]]:
        return [
            {
                "id": k,
                "proposed": self.images[k]["proposed"],
                "checked": self.images[k].get("checked"),
            }
            for k in self.sample
        ]

    def save(self, image_id: str, values: dict[str, labels.Value]) -> list[str]:
        """Write the owner's labels for one photo; what was wrong, if refused."""
        if image_id not in self.sample:
            return [f"{image_id!r} is not in the check sample"]
        values = {k: v for k, v in values.items() if not (k == "note" and v == "")}
        found = labels.problems(values, checked=True)
        if found:
            return found
        with self.lock:
            self.images[image_id] = {**self.images[image_id], "checked": values}
            labels.save(self.images, self.path)
        return []


def handler(checker: Checker) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def send(self, status: HTTPStatus, body: bytes, kind: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/":
                data = json.dumps({"rows": checker.rows(), "choices": labels.CHOICES})
                page = PAGE.read_text("utf-8").replace("/*DATA*/null", data.replace("</", "<\\/"))
                self.send(HTTPStatus.OK, page.encode("utf-8"), "text/html; charset=utf-8")
                return
            name = self.path.removeprefix("/img/").removesuffix(".png")
            if self.path.startswith("/img/") and name in checker.sample:
                picture = checker.pictures / f"{name}.png"
                if picture.exists():
                    self.send(HTTPStatus.OK, picture.read_bytes(), "image/png")
                    return
            self.send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

        def do_POST(self) -> None:
            if self.path != "/save":
                self.send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                found = checker.save(str(body["id"]), dict(body["values"]))
            except (ValueError, KeyError, TypeError) as err:
                found = [f"could not read the save: {err}"]
            status = HTTPStatus.BAD_REQUEST if found else HTTPStatus.OK
            reply = {"ok": not found, "problems": found}
            self.send(status, json.dumps(reply).encode("utf-8"), "application/json")

        def log_message(self, format: str, *args: object) -> None:
            pass  # keep the terminal quiet

    return Handler


def report(images: dict[str, dict[str, labels.Row]]) -> str:
    """How often the owner left each proposal as it was."""
    lines = [f"{name:12s} {same}/{n}" for name, (same, n) in labels.agreement(images).items()]
    return "agreement with the proposals on checked photos:\n" + "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="check the proposed labels (ADR 0004, 9b)")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--no-open", action="store_true", help="do not open the browser")
    parser.add_argument("--report", action="store_true", help="print agreement and stop")
    args = parser.parse_args(argv)
    checker = Checker()
    if args.report:
        print(report(checker.images))
        return 0
    if not checker.sample:
        print(f"no proposed labels in {checker.path}")
        return 1
    server = ThreadingHTTPServer((HOST, args.port), handler(checker))
    url = f"http://{HOST}:{args.port}/"
    print(f"check page at {url} ({len(checker.sample)} photos); Ctrl+C to stop")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
