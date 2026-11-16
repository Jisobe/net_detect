"""Download the knowledge-base documents listed in ``sources.py`` into ``data/kb/raw``."""

from __future__ import annotations

import json
import time
import urllib.parse

import requests

from .. import config
from .sources import RFCS, WEB_PAGES, WIKIPEDIA

USER_AGENT = "netdiag-kb-fetcher/0.1 (CS 746 research prototype)"


def _get(url: str, retries: int = 5, **params) -> requests.Response:
    for attempt in range(retries):
        resp = requests.get(url, params=params or None, headers={"User-Agent": USER_AGENT}, timeout=60)
        if resp.status_code != 429:
            break
        time.sleep(float(resp.headers.get("Retry-After", 2 ** (attempt + 1))))
    resp.raise_for_status()
    return resp


def fetch_all(force: bool = False) -> list[dict]:
    out_dir = config.KB_RAW_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    def save(doc_id: str, ext: str, title: str, url: str, kind: str, fetch) -> None:
        path = out_dir / f"{doc_id}.{ext}"
        if force or not path.exists():
            try:
                path.write_text(fetch(), encoding="utf-8")
                time.sleep(1.0)  # be polite (Wikipedia rate-limits bursts)
            except requests.RequestException as exc:
                print(f"  ! skipped {doc_id}: {exc}")
                return
        manifest.append({"doc_id": doc_id, "title": title, "url": url, "kind": kind, "file": path.name})
        print(f"  {doc_id}: {title}")

    for rfc, title in RFCS.items():
        url = f"https://www.rfc-editor.org/rfc/{rfc}.txt"
        save(rfc, "txt", f"{rfc.upper()}: {title}", url, "rfc", lambda u=url: _get(u).text)

    for title in WIKIPEDIA:
        doc_id = "wiki-" + title.lower().replace(" ", "-").replace("(", "").replace(")", "")
        page_url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))

        def wiki(t=title) -> str:
            data = _get("https://en.wikipedia.org/w/api.php", action="query", prop="extracts",
                        explaintext=1, redirects=1, titles=t, format="json").json()
            page = next(iter(data["query"]["pages"].values()))
            if "extract" not in page:
                raise requests.RequestException(f"no article for {t!r}")
            return page["extract"]

        save(doc_id, "txt", f"Wikipedia: {title}", page_url, "wikipedia", wiki)

    for doc_id, url in WEB_PAGES.items():
        save(doc_id, "html", doc_id.replace("-", " ").title(), url, "web", lambda u=url: _get(u).text)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
