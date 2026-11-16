"""Parse knowledge-base documents, chunk them, embed the chunks and index them in ChromaDB."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from bs4 import BeautifulSoup

from .. import config

SKIP_SECTIONS = re.compile(r"^(see also|references|external links|further reading|notes|bibliography|"
                           r"acknowledg|authors?'? address|full copyright|intellectual property|"
                           r"table of contents|contents|copyright notice|status of this memo|security considerations|"
                           r"iana considerations|normative references|informative references|"
                           r"front matter|abstract|changes from|editor'?s'? address)", re.I)
SECTION_NUMBER = re.compile(r"^(\d+(\.\d+)*\.?|Appendix [A-Z]\.?)\s+")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    url: str
    text: str

    def embedding_text(self) -> str:
        return f"{self.title} - {self.section}: {self.text}"


# --------------------------------------------------------------------------- parsers
# Each parser yields (section, paragraph) pairs.

def parse_rfc(text: str):
    lines = []
    for line in text.replace("\f", "\n").splitlines():
        if re.search(r"\[Page \d+\]\s*$", line) or re.match(r"^RFC \d+\s{2,}.*\s{2,}\w+ \d{4}\s*$", line):
            continue
        lines.append(line.rstrip())
    section, para = "Front matter", []
    for line in lines + [""]:
        heading = re.match(r"^(\d+(\.\d+)*\.?|Appendix [A-Z]\.?)\s+([A-Z].{2,80})$", line)
        if heading and not line.startswith(" "):
            if para:
                yield section, " ".join(para)
                para = []
            section = line.strip()
            continue
        if not line.strip():
            if para:
                yield section, " ".join(para)
                para = []
        else:
            para.append(line.strip())


def parse_wikipedia(text: str):
    section = "Introduction"
    for block in re.split(r"\n\s*\n|\n(?===)", text):
        block = block.strip()
        m = re.match(r"^(=+)\s*(.+?)\s*\1\s*(.*)$", block, re.S)
        if m:
            section = m.group(2)
            block = m.group(3).strip()
        if block:
            yield section, " ".join(block.split())


def parse_html(html: str):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()
    root = soup.find("main") or soup.find("article") or soup.body or soup
    section = (soup.title.string or "").strip() if soup.title else ""
    for el in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre", "dd", "dt"]):
        text = " ".join(el.get_text(" ").split())
        if el.name in ("h1", "h2", "h3", "h4"):
            section = text.rstrip("¶ ").strip()
        elif el.name == "li" and el.find(["p", "li"]):
            continue  # its children are emitted separately
        elif text:
            yield section, text


def parse_pdf(path: Path):
    from pypdf import PdfReader

    for i, page in enumerate(PdfReader(str(path)).pages, start=1):
        for para in re.split(r"\n\s*\n", page.extract_text() or ""):
            if para.strip():
                yield f"Page {i}", " ".join(para.split())


# --------------------------------------------------------------------------- chunking

def _is_noise(paragraph: str) -> bool:
    dots = paragraph.count(".") / max(len(paragraph), 1)
    return dots > 0.25 or len(paragraph.split()) < 4


def chunk_document(doc_id: str, title: str, url: str, pairs,
                   size: int = config.CHUNK_SIZE_WORDS, overlap: int = config.CHUNK_OVERLAP_WORDS) -> list[Chunk]:
    """Pack paragraphs of the same section into ~``size``-word chunks with word overlap."""
    chunks: list[Chunk] = []
    buf: list[str] = []
    current = None

    def flush():
        nonlocal buf
        if buf and len(buf) >= 25:
            chunks.append(Chunk(f"{doc_id}#{len(chunks)}", doc_id, title, current or "", url, " ".join(buf)))
        buf = buf[-overlap:] if len(buf) > overlap else []

    for section, para in pairs:
        if SKIP_SECTIONS.match(SECTION_NUMBER.sub("", section or "")):
            continue
        if section != current:
            if buf:
                flush()
            buf, current = [], section
        if _is_noise(para):
            continue
        for word in para.split():
            buf.append(word)
            if len(buf) >= size:
                flush()
    flush()
    return chunks


def load_documents() -> list[Chunk]:
    """Chunk every fetched document plus any user-supplied file in ``data/kb/local``."""
    chunks: list[Chunk] = []
    manifest_path = config.KB_RAW_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    for doc in manifest:
        raw = (config.KB_RAW_DIR / doc["file"]).read_text(encoding="utf-8")
        parser = {"rfc": parse_rfc, "wikipedia": parse_wikipedia, "web": parse_html}[doc["kind"]]
        chunks += chunk_document(doc["doc_id"], doc["title"], doc["url"], parser(raw))

    for path in sorted(config.KB_LOCAL_DIR.glob("*")):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            pairs = parse_pdf(path)
        elif suffix in (".html", ".htm"):
            pairs = parse_html(path.read_text(encoding="utf-8", errors="ignore"))
        elif suffix in (".txt", ".md"):
            pairs = parse_wikipedia(path.read_text(encoding="utf-8", errors="ignore"))
        else:
            continue
        doc_id = "local-" + re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")
        chunks += chunk_document(doc_id, path.stem, path.resolve().as_uri(), pairs)
    return chunks


def build_index(chunks: list[Chunk] | None = None) -> int:
    """Embed chunks and (re)build the Chroma collection. Returns the number of chunks."""
    from ..retrieval import VectorStore

    chunks = load_documents() if chunks is None else chunks
    if not chunks:
        raise RuntimeError("No documents found - run `netdiag fetch-kb` or add files to data/kb/local")
    config.KB_DIR.mkdir(parents=True, exist_ok=True)
    with config.KB_CHUNKS_PATH.open("w") as fh:
        for c in chunks:
            fh.write(json.dumps(asdict(c)) + "\n")
    VectorStore().rebuild(chunks)
    return len(chunks)


def read_chunks() -> list[Chunk]:
    with config.KB_CHUNKS_PATH.open() as fh:
        return [Chunk(**json.loads(line)) for line in fh]
