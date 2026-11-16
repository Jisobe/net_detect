"""Unit tests that run without downloaded data (synthetic SOFI-shaped frames)."""

import math

import numpy as np
import pandas as pd
import pytest

from netdiag import data, symptoms
from netdiag.evaluation.retrieval_eval import metrics_at_k
from netdiag.kb.ingest import chunk_document, parse_rfc, parse_wikipedia
from netdiag.retrieval import Passage, Retriever, build_query
from netdiag.taxonomy import CATEGORIES


def synthetic_sofi(n=40, device="sw", start=1_560_000_000, drop_at=None, label_at=()):
    rows = []
    for i in range(n):
        traffic = 100_000.0 if drop_at is None or i < drop_at else 300.0
        r = {"device": device, "timestamp": start + 180 * i, "range": 61,
             "ICMP_ping": 1, "ICMP_loss": 0.0, "ICMP_response_time": 0.012,
             "Device_uptime": 1000 + 165 * i, "SNMP_availability": 1,
             "class": "F" if i in label_at else "NE"}
        for itf in data.INTERFACES:
            for c in data.COUNTERS:
                r[f"{itf}_{c}"] = traffic if "Bits" in c else 0
            r[f"{itf}_Operational_status"] = 1
            r[f"{itf}_Speed"] = 1_000_000_000
            r[f"{itf}_Interface_type"] = 6
        rows.append(r)
    df = pd.DataFrame(rows)
    df["label"] = (df["class"] == "F").astype(int)
    return df


def test_clean_replaces_sentinels():
    df = synthetic_sofi(5)
    df.loc[0, "ICMP_ping"] = -1
    df.loc[1, "IN_Bits_received"] = -2
    df.loc[2, "IN_Operational_status"] = 0
    df.loc[3, "ICMP_loss"] = 666.667
    out = data.clean(df)
    assert math.isnan(out.loc[0, "ICMP_ping"])
    assert math.isnan(out.loc[1, "IN_Bits_received"])
    assert math.isnan(out.loc[2, "IN_Operational_status"])
    assert math.isnan(out.loc[3, "ICMP_loss"])
    assert out.loc[4, "EX_Bits_sent"] == 100_000


def test_features_are_causal():
    """Changing a future sample must not change features of earlier samples."""
    a = data.add_features(data.clean(synthetic_sofi(30)))
    b_raw = synthetic_sofi(30)
    b_raw.loc[25:, "EX_Bits_received"] = 5
    b = data.add_features(data.clean(b_raw))
    cols = data.feature_columns(a)
    pd.testing.assert_frame_equal(a.loc[:24, cols], b.loc[:24, cols])


def test_traffic_drop_is_visible_in_features():
    df = data.add_features(data.clean(synthetic_sofi(30, drop_at=20)))
    assert df.loc[21, "total_traffic__log_ratio"] < -3
    assert abs(df.loc[15, "total_traffic__log_ratio"]) < 1e-9


def test_rolling_baseline_resets_on_gap():
    df = synthetic_sofi(20)
    df.loc[10:, "timestamp"] += 10_000  # collection gap
    out = data.add_features(data.clean(df))
    assert out.loc[10, "segment"] != out.loc[9, "segment"]
    assert math.isnan(out.loc[10, "EX_Bits_received__baseline"])


def test_splits_keep_time_blocks_together():
    frames = [synthetic_sofi(480, device=d, label_at=set(range(0, 480, 37)), start=1_560_000_000)
              for d in ("a", "b")]
    df = pd.concat(frames, ignore_index=True)
    split = data.assign_splits(df, block_hours=1)
    block = df["timestamp"] // 3600
    assert (pd.crosstab(block, split) > 0).sum(axis=1).max() == 1
    assert set(split) == {"train", "val", "test"}


def test_profiler_detects_traffic_loss():
    df = data.add_features(data.clean(synthetic_sofi(30, drop_at=20)))
    prof = symptoms.profile(df.loc[22], 0.9)
    assert prof.category == "traffic_loss"
    assert "Traffic loss" in prof.description
    assert "probability 0.90" in prof.description


def test_profiler_detects_link_down_and_errors():
    df = data.add_features(data.clean(synthetic_sofi(30)))
    row = df.loc[25].copy()
    row["EX_Operational_status"] = 2
    assert symptoms.profile(row).category == "link_down"
    row = df.loc[25].copy()
    row["IN_Inbound_packets_with_errors"] = 40
    row["IN_Inbound_packets_with_errors__excess"] = 40
    assert symptoms.profile(row).category == "interface_errors"


def test_profiler_defaults_to_unclassified():
    df = data.add_features(data.clean(synthetic_sofi(30)))
    assert symptoms.profile(df.loc[25]).category == "unclassified"


def test_metrics_at_k():
    m = metrics_at_k(["a", "x", "b", "y"], {"a", "b", "c", "d"}, ks=(1, 3))
    assert m["P@1"] == 1.0 and m["R@1"] == 0.25
    assert m["P@3"] == pytest.approx(2 / 3) and m["R@3"] == 0.5
    assert m["MRR"] == 1.0
    assert metrics_at_k(["x"], {"a"}, ks=(1,))["MRR"] == 0.0


def test_rfc_parser_and_chunker_skip_boilerplate():
    text = ("1.  Introduction\n\n" + "The ifOperStatus object reports the state. " * 40 +
            "\n\nRFC 2863          The Interfaces Group MIB            June 2000\n"
            "McCloghrie & Kastenholz     Standards Track                    [Page 3]\n\n"
            "8.  References\n\n" + "[1] Some reference text here please. " * 20)
    chunks = chunk_document("rfc", "RFC", "u", parse_rfc(text), size=100, overlap=10)
    assert chunks and all(c.section == "1.  Introduction" for c in chunks)
    assert not any("Page 3" in c.text for c in chunks)


def test_wikipedia_parser_sections():
    pairs = list(parse_wikipedia("Intro text here.\n\n== Causes ==\nCable faults cause errors.\n\n== See also ==\nX"))
    assert ("Causes", "Cable faults cause errors.") in pairs


class FakeStore:
    """Stands in for the Chroma vector store: returns canned passages per query, no embeddings."""

    def __init__(self, first_score, expansion_hits=()):
        self.first_score = first_score
        self.expansion_hits = list(expansion_hits)
        self.queries = []

    def query(self, text, k):
        self.queries.append(text)
        if len(self.queries) == 1:
            return [Passage("orig", "Doc", "Sec", "u", "text", self.first_score)]
        return self.expansion_hits


def test_build_query_uses_category_and_evidence():
    q = build_query("traffic_loss", ["traffic down 99%", "all interfaces up"])
    assert q.startswith(CATEGORIES["traffic_loss"].title)
    assert "traffic down 99%; all interfaces up" in q


def test_retriever_skips_reformulation_when_confident():
    store = FakeStore(first_score=0.9)
    res = Retriever(store, k=5, threshold=0.45).retrieve("traffic_loss", ["x"], alt_categories=["link_down"])
    assert not res.reformulated and len(store.queries) == 1
    assert res.confidence == 0.9


def test_retriever_reformulates_and_merges_when_low_confidence():
    better = Passage("better", "Doc", "Sec", "u", "text", 0.7)
    store = FakeStore(first_score=0.2, expansion_hits=[better])
    res = Retriever(store, k=5, threshold=0.45).retrieve("traffic_loss", ["x"],
                                                         alt_categories=["traffic_loss", "link_down"])
    # one expansion for the category itself, one for the distinct alternative
    assert res.reformulated and len(store.queries) == 3
    assert CATEGORIES["link_down"].query_terms in store.queries[-1]
    assert [p.chunk_id for p in res.passages] == ["better", "orig"]


def test_retriever_agentic_flag_disables_reformulation():
    store = FakeStore(first_score=0.1, expansion_hits=[Passage("b", "D", "S", "u", "t", 0.8)])
    res = Retriever(store, threshold=0.45).retrieve("traffic_loss", ["x"], agentic=False)
    assert not res.reformulated and len(store.queries) == 1


def test_feature_columns_exclude_identifiers():
    df = data.add_features(data.clean(synthetic_sofi(10)))
    cols = data.feature_columns(df)
    assert not {"timestamp", "range", "label", "device", "Device_uptime"} & set(cols)
    assert np.isfinite(df[cols].select_dtypes("number").fillna(0).to_numpy()).all()
