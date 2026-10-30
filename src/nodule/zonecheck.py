"""The owner's check of the proposed C23-C25 (ADR 0006, sections 3 and 4).

`python -m nodule.zonecheck` serves a page on this PC only (127.0.0.1), like
`nodule.check`, over the same 30-photo check sample and outline-check pictures.
The `zones.BLIND` blind faces come first, with the proposals hidden; the rest
can be saved only once those are done (choice 7f). C24's cues start blank on
every face, so they are answered unprompted. Saves are checked
(`zones.problems`) and written to `references/band-zone-labels.toml`.

`--recount` serves the delayed eye recount instead (choice 7h): `zones.RECOUNT`
faces the owner gave a band count on, from `zones.RECOUNT_FROM`, without the
earlier count. Nothing leaves the PC.
"""

from __future__ import annotations

import argparse
import threading
import webbrowser
from collections.abc import Sequence
from datetime import date
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

from nodule import labels, zones
from nodule.check import HOST, handler
from nodule.overlay import OUT_DIR

PORT = 8766


class ZoneChecker:
    """The sample, its blind part and the C23-C25 labels file."""

    page = Path(__file__).with_name("zonecheck.html")

    def __init__(
        self,
        path: Path = zones.ZONES_TOML,
        pictures: Path = OUT_DIR,
        view_labels: Path = labels.LABELS_TOML,
        recount: bool = False,
        today: date | None = None,
    ) -> None:
        self.path = path
        self.pictures = pictures
        self.recount = recount
        self.today = today or date.today()
        self.images = zones.load(path)
        views = labels.load(view_labels)
        sample = labels.check_sample(views)
        self.blind = zones.blind_sample(sample)
        if recount:
            self.sample = zones.recount_sample(views)
        else:
            have = [k for k in sample if "proposed" in self.images.get(k, {})]
            self.sample = [k for k in have if k in self.blind] + [
                k for k in have if k not in self.blind
            ]
        self.lock = threading.Lock()

    def data(self) -> dict[str, object]:
        return {
            "mode": "recount" if self.recount else "zones",
            "open": not self.recount or self.today >= zones.RECOUNT_FROM,
            "opens": zones.RECOUNT_FROM.isoformat(),
            "rows": self.rows(),
            "kinds": zones.ZONE_KINDS,
            "thirds": zones.THIRDS,
            "cues": zones.CUES,
            "cue_values": zones.CUE_VALUES,
        }

    def rows(self) -> list[dict[str, object]]:
        if self.recount:
            return [
                {"id": k, "recount": self.images.get(k, {}).get("recount")} for k in self.sample
            ]
        out: list[dict[str, object]] = []
        for k in self.sample:
            blind = k in self.blind
            proposed = {
                n: v for n, v in self.images[k]["proposed"].items() if n not in zones.CUE_FIELDS
            }
            out.append(
                {
                    "id": k,
                    "blind": blind,
                    "proposed": None if blind else proposed,
                    "checked": self.images[k].get("checked"),
                }
            )
        return out

    def picture(self, name: str) -> Path | None:
        return self.pictures / f"{name}.png" if name in self.sample else None

    def save(self, image_id: str, values: dict[str, Any]) -> list[str]:
        """Write the owner's answer for one face; what was wrong, if refused."""
        if image_id not in self.sample:
            return [f"{image_id!r} is not in the sample"]
        if self.recount:
            return self.save_recount(image_id, values)
        if image_id not in self.blind:
            left = [k for k in self.sample if k in self.blind and "checked" not in self.images[k]]
            if left:
                return [f"answer the blind faces first ({len(left)} left)"]
        values = {k: v for k, v in values.items() if not (k == "note" and v == "")}
        values["blind"] = image_id in self.blind
        found = zones.problems(values, checked=True)
        if found:
            return found
        with self.lock:
            self.images[image_id] = {**self.images[image_id], "checked": values}
            zones.save(self.images, self.path)
        return []

    def save_recount(self, image_id: str, values: dict[str, Any]) -> list[str]:
        if self.today < zones.RECOUNT_FROM:
            return [f"the recount opens on {zones.RECOUNT_FROM.isoformat()}"]
        count = values.get("C6_eye")
        low, high = labels.EYE_COUNT
        if count != zones.UNKNOWN and (
            isinstance(count, bool) or not isinstance(count, int) or not low <= count <= high
        ):
            return [f"C6_eye: {count!r} is not a whole number from {low} to {high}, or unknown"]
        note = values.get("note", "")
        if not isinstance(note, str):
            return ["note: not text"]
        recount = {"C6_eye": count, "date": self.today.isoformat()} | (
            {"note": note} if note else {}
        )
        with self.lock:
            entry = self.images.get(image_id, {})
            self.images[image_id] = {**entry, "recount": recount}
            zones.save(self.images, self.path)
        return []


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="check the proposed C23-C25 (ADR 0006)")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--no-open", action="store_true", help="do not open the browser")
    parser.add_argument("--recount", action="store_true", help="the delayed eye recount (7h)")
    args = parser.parse_args(argv)
    checker = ZoneChecker(recount=args.recount)
    if not checker.sample:
        print(f"nothing to check: no proposals in {checker.path}")
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
