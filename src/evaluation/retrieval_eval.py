"""Retrieval quality: Precision@k, Recall@k, Hit@k and MRR.

Relevance judgments are produced automatically from each category's relevance
patterns in ``taxonomy.py`` (a chunk is relevant to a category when it matches at
least ``min_matches`` distinct patterns). This is a transparent, reproducible proxy
for human judgments; the qrels are written to disk so they can be audited or
replaced with hand labels.
"""

from __future__ import annotations

import json
import re

import pandas as pd

from .. import config, data
from ..classifier import FaultDetector
from ..kb.ingest import read_chunks
from ..retrieval import Retriever, VectorStore
from ..symptoms import profile
from ..taxonomy import CATEGORIES

CANONICAL_QUERIES = {
    "link_down": [
        "Switch interface shows operational status down while admin status is up",
        "Uplink port went down, link light off, ifOperStatus lowerLayerDown",
        "Link flapping on core switch port, carrier transitions increasing",
    ],
    "device_unreachable": [
        "Core switch stopped answering ping and SNMP polls time out",
        "ICMP echo requests to the switch management address get no reply",
        "Monitoring lost the device: SNMP agent unreachable, 100% packet loss",
    ],
    "device_restart": [
        "sysUpTime went back to zero, the switch rebooted unexpectedly",
        "Device uptime reset and a coldStart trap was received",
        "Switch reloaded overnight, need to find out why it restarted",
    ],
    "traffic_loss": [
        "Interfaces are up but traffic through the switch dropped to almost zero",
        "Packets are being black-holed, throughput collapsed with no errors",
        "Users lost connectivity although links are up; forwarding seems broken",
    ],
    "interface_errors": [
        "Input errors and CRC errors increasing on a switch port",
        "FCS errors and late collisions suggest a duplex mismatch",
        "Suspected bad cable: alignment errors and runts on the interface",
    ],
    "packet_discards": [
        "Output discards rising on an uplink during peak hours",
        "Interface utilization near line rate and packets are being dropped",
        "Buffer overflow and queue drops on a congested link",
    ],
    "high_latency": [
        "Ping round-trip time to the switch increased tenfold",
        "High latency and jitter across the core, queueing delay suspected",
        "Response times degraded although there is no packet loss",
    ],
}


def build_qrels(chunks) -> dict[str, set[str]]:
    qrels = {}
    for key, cat in CATEGORIES.items():
        pats = [re.compile(p, re.I) for p in cat.relevance]
        qrels[key] = {c.chunk_id for c in chunks
                      if sum(bool(p.search(f"{c.section} {c.text}")) for p in pats) >= cat.min_matches}
    return qrels


def metrics_at_k(retrieved: list[str], relevant: set[str], ks=(1, 3, 5, 10)) -> dict:
    out = {}
    for k in ks:
        hits = sum(1 for cid in retrieved[:k] if cid in relevant)
        out[f"P@{k}"] = hits / k
        out[f"R@{k}"] = hits / len(relevant) if relevant else 0.0
        out[f"Hit@{k}"] = float(hits > 0)
    rank = next((i for i, cid in enumerate(retrieved, start=1) if cid in relevant), None)
    out["MRR"] = 1.0 / rank if rank else 0.0
    return out


def sofi_queries(max_per_category: int = 15) -> list[dict]:
    """Queries built from detected faults in the SOFI test split, as the pipeline would."""
    df = data.load_dataset()
    df = df[data.assign_splits(df) == "test"]
    det = FaultDetector.load()
    df = df.assign(p=det.predict_proba(df))
    flagged = df[(df["p"] >= det.threshold) & (df["label"] == 1)]
    out, counts = [], {}
    for _, row in flagged.iterrows():
        prof = profile(row, row["p"])
        cat = prof.category
        if counts.get(cat, 0) >= max_per_category:
            continue
        counts[cat] = counts.get(cat, 0) + 1
        ranked = prof.ranked()
        out.append({"category": cat, "evidence": [e for k, _ in ranked for e in prof.evidence.get(k, [])],
                    "alt": [k for k, _ in ranked[1:3]], "source": "sofi-test"})
    return out


def evaluate_retrieval(k: int = 10, include_sofi: bool = True) -> dict:
    chunks = read_chunks()
    qrels = build_qrels(chunks)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "qrels.json").write_text(json.dumps({c: sorted(v) for c, v in qrels.items()}, indent=1))

    queries = [{"category": c, "text": q, "source": "canonical"}
               for c, qs in CANONICAL_QUERIES.items() for q in qs]
    if include_sofi:
        queries += sofi_queries()

    store = VectorStore()
    retriever = Retriever(store, k=k)
    rows = []
    for q in queries:
        for mode in ("single-shot", "agentic"):
            if q["source"] == "canonical":
                text = q["text"]
                if mode == "single-shot":
                    passages = store.query(text, k)
                else:
                    res = retriever.retrieve(q["category"], [text])
                    passages = res.passages
            else:
                res = retriever.retrieve(q["category"], q["evidence"], alt_categories=q["alt"],
                                         agentic=(mode == "agentic"))
                passages = res.passages
            m = metrics_at_k([p.chunk_id for p in passages], qrels[q["category"]])
            rows.append({"category": q["category"], "source": q["source"], "mode": mode, **m})

    df = pd.DataFrame(rows)
    cols = [c for c in df.columns if c[0] in "PRHM"]
    summary = df.groupby(["source", "mode"])[cols].mean().round(3)
    by_cat = df[df["mode"] == "agentic"].groupby("category")[cols].mean().round(3)
    result = {
        "n_queries": int(len(df) / 2),
        "relevant_chunks_per_category": {c: len(v) for c, v in qrels.items()},
        "summary": summary.reset_index().to_dict(orient="records"),
        "by_category_agentic": by_cat.reset_index().to_dict(orient="records"),
    }
    (config.REPORTS_DIR / "retrieval_metrics.json").write_text(json.dumps(result, indent=2))
    md = ["# Retrieval evaluation", "",
          f"{result['n_queries']} queries; relevance = category keyword patterns (see `qrels.json`).", "",
          "## Overall", "", summary.reset_index().to_markdown(index=False), "",
          "## By category (agentic mode)", "", by_cat.reset_index().to_markdown(index=False), "",
          "Relevant chunks per category: " + ", ".join(f"{c}={n}" for c, n in
                                                       result["relevant_chunks_per_category"].items())]
    (config.REPORTS_DIR / "retrieval_metrics.md").write_text("\n".join(md) + "\n")
    return result
