"""Symptom profiling: turn one featurized SOFI sample into fault-category evidence
and a natural-language description that is used as the retrieval query."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from .data import ERROR_COLS, INTERFACES, STATUS_COLS, TRAFFIC_COLS
from .taxonomy import CATEGORIES

IF_NAMES = {"P": "port P", "EX": "external interface EX", "IN": "internal interface IN"}
OPER_STATUS = {1: "up", 2: "down", 3: "testing", 4: "unknown", 5: "dormant", 6: "notPresent", 7: "lowerLayerDown"}


def _ok(v) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def fmt_bps(v: float) -> str:
    if not _ok(v):
        return "n/a"
    for unit, scale in (("Gb/s", 1e9), ("Mb/s", 1e6), ("kb/s", 1e3)):
        if abs(v) >= scale:
            return f"{v / scale:.1f} {unit}"
    return f"{v:.0f} b/s"


@dataclass
class SymptomProfile:
    scores: dict[str, float]
    evidence: dict[str, list[str]] = field(default_factory=dict)
    description: str = ""

    @property
    def category(self) -> str:
        return max(self.scores, key=self.scores.get)

    def ranked(self, min_score: float = 0.3) -> list[tuple[str, float]]:
        items = sorted(self.scores.items(), key=lambda kv: -kv[1])
        return [kv for kv in items if kv[1] >= min_score] or items[:1]


def score_categories(row: pd.Series) -> tuple[dict[str, float], dict[str, list[str]]]:
    """Heuristic evidence score in [0, 1] for each fault category."""
    s = {k: 0.0 for k in CATEGORIES}
    ev: dict[str, list[str]] = {k: [] for k in CATEGORIES}

    for col in STATUS_COLS:
        v = row.get(col)
        if _ok(v) and v != 1:
            s["link_down"] = 1.0
            ev["link_down"].append(f"{IF_NAMES[col.split('_')[0]]} ifOperStatus={OPER_STATUS.get(int(v), v)}")

    loss = row.get("ICMP_loss")
    if _ok(loss) and loss > 0:
        s["device_unreachable"] = max(s["device_unreachable"], min(1.0, loss / 100))
        ev["device_unreachable"].append(f"ICMP loss {loss:.0f}%")
    if row.get("SNMP_availability") == 0:
        s["device_unreachable"] = 1.0
        ev["device_unreachable"].append("SNMP agent not responding")

    if row.get("uptime_reset") == 1:
        s["device_restart"] = 0.9
        ev["device_restart"].append("sysUpTime decreased since previous poll")

    drop = row.get("total_traffic__log_ratio")
    if _ok(drop) and drop < -1:
        s["traffic_loss"] = min(1.0, -drop / 3)
        ev["traffic_loss"].append(f"total traffic {100 * (math.exp(drop) - 1):.0f}% vs 30-min baseline")
    for col in ("fwd_ratio_ex_to_in", "fwd_ratio_in_to_ex"):
        v = row.get(col)
        if _ok(v) and abs(v) > 1.5:
            s["traffic_loss"] = max(s["traffic_loss"], min(1.0, abs(v) / 4))
            ev["traffic_loss"].append(f"forwarding imbalance {col}={v:+.1f} (log ratio)")

    for col in ("EX_Inbound_packets_with_errors", "IN_Inbound_packets_with_errors"):
        excess = row.get(f"{col}__excess")
        if _ok(excess) and excess > 3:
            s["interface_errors"] = max(s["interface_errors"], min(1.0, excess / 10))
            ev["interface_errors"].append(f"{col}={row[col]:.0f} ({excess:+.1f} above baseline)")
    for col in ERROR_COLS:
        v = row.get(col)
        if "discarded" in col and _ok(v) and v > 0:
            s["packet_discards"] = max(s["packet_discards"], min(1.0, 0.5 + v / 20))
            ev["packet_discards"].append(f"{col}={v:.0f}")
        if "Outbound_packets_with_errors" in col and _ok(v) and v > 0:
            s["interface_errors"] = max(s["interface_errors"], 0.6)
            ev["interface_errors"].append(f"{col}={v:.0f}")
    for i in INTERFACES:
        speed = row.get(f"{i}_Speed")
        peak = max((row.get(f"{i}_{d}") or 0) for d in ("Bits_received", "Bits_sent"))
        if _ok(speed) and speed > 0 and _ok(peak) and peak / speed > 0.7:
            s["packet_discards"] = max(s["packet_discards"], min(1.0, peak / speed))
            ev["packet_discards"].append(f"{IF_NAMES[i]} at {100 * peak / speed:.0f}% of line rate")

    rtt = row.get("ICMP_response_time__ratio")
    if _ok(rtt) and rtt > 3:
        s["high_latency"] = min(1.0, rtt / 8)
        ev["high_latency"].append(f"ICMP RTT {rtt:.1f}x baseline")

    s["unclassified"] = 0.25
    return s, ev


def describe(row: pd.Series, scores: dict[str, float], evidence: dict[str, list[str]],
             fault_probability: float | None = None) -> str:
    """Natural-language symptom summary for one sample (the retrieval query body)."""
    when = pd.to_datetime(row["timestamp"], unit="s").strftime("%Y-%m-%d %H:%M UTC")
    parts = [f"Core switch {row['device']} at {when}."]
    if fault_probability is not None:
        parts.append(f"Fault detector probability {fault_probability:.2f}.")

    status = []
    for i in INTERFACES:
        v = row.get(f"{i}_Operational_status")
        state = OPER_STATUS.get(int(v), str(v)) if _ok(v) else "not reported"
        speed = row.get(f"{i}_Speed")
        status.append(f"{IF_NAMES[i]} {state}" + (f" at {fmt_bps(speed)}" if _ok(speed) else ""))
    parts.append("Interface status: " + "; ".join(status) + ".")

    traffic = []
    for col in TRAFFIC_COLS:
        v, base = row.get(col), row.get(f"{col}__baseline")
        if not _ok(v):
            continue
        txt = f"{col.replace('_', ' ')} {fmt_bps(v)}"
        if _ok(base) and base > 0:
            txt += f" (baseline {fmt_bps(base)}, {100 * (v - base) / base:+.0f}%)"
        traffic.append(txt)
    parts.append("Traffic: " + ("; ".join(traffic) if traffic else "counters not collected") + ".")

    errs = [f"{c.replace('_', ' ')} {row[c]:.0f}" for c in ERROR_COLS if _ok(row.get(c)) and row[c] > 0]
    parts.append("Errors/discards: " + ("; ".join(errs) if errs else "none") + ".")

    if _ok(row.get("ICMP_response_time")):
        rtt = f"ICMP echo reply in {1000 * row['ICMP_response_time']:.1f} ms"
        if _ok(row.get("ICMP_response_time__baseline")):
            rtt += f" (baseline {1000 * row['ICMP_response_time__baseline']:.1f} ms)"
        loss = row.get("ICMP_loss")
        parts.append(rtt + (f", {loss:.0f}% loss." if _ok(loss) else "."))
    else:
        parts.append("ICMP probe result not collected.")
    if _ok(row.get("Device_uptime")):
        restart = " - uptime counter reset since previous poll" if row.get("uptime_reset") == 1 else ""
        parts.append(f"Device uptime counter {row['Device_uptime']:.0f}{restart}.")

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    top = [k for k, v in ranked if v >= 0.3] or [ranked[0][0]]
    pattern = ", ".join(CATEGORIES[k].title for k in top)
    parts.append(f"Observed symptom pattern: {pattern}.")
    details = [e for k in top for e in evidence.get(k, [])]
    if details:
        parts.append("Evidence: " + "; ".join(details) + ".")
    return " ".join(parts)


def profile(row: pd.Series, fault_probability: float | None = None) -> SymptomProfile:
    scores, evidence = score_categories(row)
    return SymptomProfile(scores, evidence, describe(row, scores, evidence, fault_probability))
