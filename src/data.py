"""Load, clean and featurize the SOFI (Symptom-Fault relationship for IP-Network) dataset.

SOFI ships two CSVs, one per core switch, sampled every 180 s over SNMP and ICMP.
Column prefixes: ``P_`` / ``EX_`` / ``IN_`` are three monitored interfaces of the
switch (port, external-facing and internal-facing); ``ICMP_*`` is the reachability
probe; ``Device_uptime`` and ``SNMP_availability`` describe the switch itself.
The label column ``class`` is binary: ``NE`` (no error) or ``F`` (fault).

Cleaning decisions (see README for rationale):
* ``-1`` (and ``-2`` on ``IN_*`` counters, ``0`` on ``IN_Operational_status``) are
  "value not collected" sentinels, not measurements -> NaN plus a missing flag.
* ``ICMP_loss`` values above 100 % are collector glitches -> NaN.
* ``timestamp`` and ``range`` are not used as model inputs (identifier-like).
* Temporal features only look at *past* samples, so the detector can run online.
"""

from __future__ import annotations

import hashlib
import io
import urllib.request

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from . import config

INTERFACES = ("P", "EX", "IN")
COUNTERS = (
    "Inbound_packets_discarded",
    "Inbound_packets_with_errors",
    "Bits_received",
    "Outbound_packets_discarded",
    "Outbound_packets_with_errors",
    "Bits_sent",
)
TRAFFIC_COLS = [f"{i}_{c}" for i in INTERFACES for c in ("Bits_received", "Bits_sent")]
ERROR_COLS = [f"{i}_{c}" for i in INTERFACES for c in COUNTERS if "errors" in c or "discarded" in c]
STATUS_COLS = [f"{i}_Operational_status" for i in INTERFACES]
SPEED_COLS = [f"{i}_Speed" for i in INTERFACES]
TYPE_COLS = [f"{i}_Interface_type" for i in INTERFACES]
ICMP_COLS = ["ICMP_ping", "ICMP_loss", "ICMP_response_time"]
DEVICE_COLS = ["Device_uptime", "SNMP_availability"]
RAW_FEATURE_COLS = ICMP_COLS + TRAFFIC_COLS + ERROR_COLS + STATUS_COLS + DEVICE_COLS


# --------------------------------------------------------------------------- loading

def download_sofi(force: bool = False) -> None:
    """Download both SOFI CSVs from Mendeley Data and verify their SHA-256."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    for device, meta in config.SOFI_FILES.items():
        path = config.RAW_DIR / meta["filename"]
        if path.exists() and not force:
            continue
        print(f"Downloading SOFI {device} ...")
        # Mendeley rejects urllib's default User-Agent with 403, so send an explicit one.
        req = urllib.request.Request(meta["url"], headers={"User-Agent": "netdiag/0.1"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = resp.read()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != meta["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch for {device}: {digest}")
        path.write_bytes(payload)


def load_raw() -> pd.DataFrame:
    """Return both switches stacked, with a ``device`` column and a binary ``label``."""
    frames = []
    for device, meta in config.SOFI_FILES.items():
        path = config.RAW_DIR / meta["filename"]
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - run `netdiag download-data` first")
        # The files use bare CR line endings.
        text = path.read_text().replace("\r\n", "\n").replace("\r", "\n")
        df = pd.read_csv(io.StringIO(text))
        df = df.rename(columns={"SNMP_Availability": "SNMP_availability"})
        df.insert(0, "device", device)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["label"] = (df["class"] == "F").astype(int)
    return df


# --------------------------------------------------------------------------- cleaning

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Replace collector sentinels with NaN and drop impossible values."""
    df = df.copy()
    sentinel_cols = ICMP_COLS + TRAFFIC_COLS + ERROR_COLS + STATUS_COLS + SPEED_COLS + TYPE_COLS + DEVICE_COLS
    for col in sentinel_cols:
        df[col] = df[col].astype(float).where(df[col] >= 0)
    # IN_Operational_status reports 0 exactly when the IN_* counters are missing;
    # 0 is not a valid ifOperStatus value (RFC 2863), so treat it as missing too.
    df.loc[df["IN_Operational_status"] == 0, "IN_Operational_status"] = np.nan
    df.loc[df["ICMP_loss"] > 100, "ICMP_loss"] = np.nan
    return df


# --------------------------------------------------------------------------- features

def _segments(ts: pd.Series) -> pd.Series:
    """Contiguous runs of samples spaced exactly one polling interval apart."""
    return (ts.diff() != config.SAMPLE_INTERVAL_S).cumsum()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add causal temporal features computed per device and per contiguous segment."""
    df = df.sort_values(["device", "timestamp"]).reset_index(drop=True)
    df["segment"] = df.groupby("device")["timestamp"].transform(_segments)
    g = df.groupby(["device", "segment"], sort=False)
    w = config.BASELINE_WINDOW

    def past(col: str, fn: str) -> pd.Series:
        return g[col].transform(lambda s: getattr(s.shift(1).rolling(w, min_periods=3), fn)())

    feats = {}
    for col in TRAFFIC_COLS:
        base = past(col, "median")
        feats[f"{col}__log"] = np.log1p(df[col])
        feats[f"{col}__baseline"] = base
        feats[f"{col}__log_ratio"] = np.log1p(df[col]) - np.log1p(base)
        feats[f"{col}__log_delta"] = np.log1p(df[col]) - np.log1p(g[col].shift(1))
    for col in ("EX_Inbound_packets_with_errors", "IN_Inbound_packets_with_errors"):
        feats[f"{col}__baseline"] = past(col, "mean")
        feats[f"{col}__excess"] = df[col] - feats[f"{col}__baseline"]
    rtt_base = past("ICMP_response_time", "median")
    feats["ICMP_response_time__baseline"] = rtt_base
    feats["ICMP_response_time__ratio"] = df["ICMP_response_time"] / rtt_base

    # Traffic forwarded through the switch: what arrives on EX should leave on IN.
    feats["fwd_ratio_ex_to_in"] = np.log1p(df["IN_Bits_sent"]) - np.log1p(df["EX_Bits_received"])
    feats["fwd_ratio_in_to_ex"] = np.log1p(df["EX_Bits_sent"]) - np.log1p(df["IN_Bits_received"])
    feats["total_traffic__log_ratio"] = np.log1p(df[TRAFFIC_COLS].sum(axis=1, min_count=1)) - np.log1p(
        sum(feats[f"{c}__baseline"].fillna(0) for c in TRAFFIC_COLS)
    )
    # Faults persist for several polls, so short-term memory of the drop helps.
    short = pd.DataFrame({k: feats[k] for k in ("total_traffic__log_ratio", "fwd_ratio_ex_to_in", "fwd_ratio_in_to_ex")})
    short[["device", "segment"]] = df[["device", "segment"]]
    sg = short.groupby(["device", "segment"], sort=False)
    for k in ("total_traffic__log_ratio", "fwd_ratio_ex_to_in", "fwd_ratio_in_to_ex"):
        feats[f"{k}__lag1"] = sg[k].shift(1)
        feats[f"{k}__mean3"] = sg[k].transform(lambda s: s.rolling(3, min_periods=1).mean())
    feats["uptime_reset"] = (g["Device_uptime"].diff() < 0).astype(float)

    feats["missing_icmp"] = df["ICMP_ping"].isna().astype(float)
    feats["missing_snmp"] = df["SNMP_availability"].isna().astype(float)
    feats["missing_p_ex_counters"] = df["EX_Bits_received"].isna().astype(float)
    feats["missing_in_counters"] = df["IN_Bits_received"].isna().astype(float)
    feats["missing_baseline"] = feats["EX_Bits_received__baseline"].isna().astype(float)

    return pd.concat([df, pd.DataFrame(feats)], axis=1)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model inputs: cleaned raw measurements plus engineered features."""
    raw = [c for c in RAW_FEATURE_COLS if c not in ("Device_uptime",)]  # absolute uptime is spurious
    eng = [c for c in df.columns if "__" in c or c.startswith(("fwd_", "missing_", "uptime_"))]
    return raw + eng


def load_dataset() -> pd.DataFrame:
    """Cleaned and featurized SOFI data for both switches."""
    return add_features(clean(load_raw()))


# --------------------------------------------------------------------------- splits

def assign_splits(df: pd.DataFrame, block_hours: int = 24, seed: int = config.RANDOM_SEED) -> pd.Series:
    """Stratified train/val/test split (~70/15/15) over whole time blocks.

    Adjacent samples of one fault episode are near-duplicates, and both switches
    observe the same episodes, so rows are grouped into time blocks shared across
    devices; a block is never split between partitions.
    """
    groups = (df["timestamp"] // (block_hours * 3600)).to_numpy()
    y = df["label"].to_numpy()
    split = np.empty(len(df), dtype=object)

    outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    rest_idx, test_idx = next(outer.split(np.zeros(len(df)), y, groups))
    split[test_idx] = "test"

    inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed + 1)
    tr, va = next(inner.split(np.zeros(len(rest_idx)), y[rest_idx], groups[rest_idx]))
    split[rest_idx[tr]] = "train"
    split[rest_idx[va]] = "val"
    return pd.Series(split, index=df.index, name="split")


def fault_episodes(df: pd.DataFrame) -> pd.Series:
    """Episode id for each fault row (consecutive fault samples on one device), else -1."""
    df = df.sort_values(["device", "timestamp"])
    new = (df["label"] == 1) & ((df["label"].shift(1) != 1) | (df["device"] != df["device"].shift(1)))
    ep = new.cumsum().where(df["label"] == 1, -1)
    return ep.reindex(df.index)
