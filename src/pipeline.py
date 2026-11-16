"""Network diagnosis assistant: detector -> symptom profile -> retrieval of troubleshooting passages."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd

from . import config
from .classifier import FaultDetector
from .retrieval import RetrievalResult, Retriever
from .symptoms import SymptomProfile, profile
from .taxonomy import CATEGORIES


@dataclass
class DiagnosisReport:
    device: str
    timestamp: int
    fault_probability: float
    is_fault: bool
    profile: SymptomProfile
    retrieval: RetrievalResult | None = None
    timings: dict[str, float] = field(default_factory=dict)
    true_label: int | None = None

    def classifier_only(self) -> str:
        """What stage 1 alone tells the engineer (the RAG ablation baseline)."""
        verdict = "FAULT" if self.is_fault else "no fault"
        return f"{self.device} @ {self.timestamp}: {verdict} (p={self.fault_probability:.2f})"

    def to_markdown(self) -> str:
        when = pd.to_datetime(self.timestamp, unit="s").strftime("%Y-%m-%d %H:%M UTC")
        lines = [f"# Diagnosis - {self.device} at {when}", "",
                 f"**Detector:** {'FAULT' if self.is_fault else 'no fault'} "
                 f"(probability {self.fault_probability:.2f})"]
        if self.true_label is not None:
            lines.append(f"**SOFI label:** {'F (fault)' if self.true_label else 'NE (no error)'}")
        ranked = ", ".join(f"{CATEGORIES[k].title} ({v:.2f})" for k, v in self.profile.ranked())
        lines += [f"**Symptom pattern:** {ranked}", "", "## Telemetry summary", self.profile.description, ""]
        if self.retrieval:
            lines += ["## Sources"]
            lines += [f"[{i}] {p.citation()} - {p.url} (similarity {p.score:.2f})"
                      for i, p in enumerate(self.retrieval.passages, start=1)]
            if self.retrieval.reformulated:
                lines += ["", "_Retrieval confidence was low, so the query was reformulated: "
                          + "; ".join(self.retrieval.trace) + "_"]
        return "\n".join(lines)


class DiagnosisAssistant:
    def __init__(self, detector: FaultDetector | None = None,
                 retriever: Retriever | None = None, agentic: bool = True):
        self.detector = detector or FaultDetector.load()
        self.retriever = retriever or Retriever()
        self.agentic = agentic

    def diagnose(self, row: pd.Series, always_explain: bool = False) -> DiagnosisReport:
        """Diagnose one featurized SOFI sample (a row from ``data.load_dataset()``)."""
        t0 = time.perf_counter()
        frame = row.to_frame().T.astype({c: float for c in self.detector.features})
        p = float(self.detector.predict_proba(frame)[0])
        is_fault = p >= self.detector.threshold
        prof = profile(row, p)
        report = DiagnosisReport(row["device"], int(row["timestamp"]), p, is_fault, prof,
                                 true_label=int(row["label"]) if "label" in row else None)
        report.timings["classify_s"] = time.perf_counter() - t0
        if not (is_fault or always_explain):
            return report

        t1 = time.perf_counter()
        ranked = prof.ranked()
        evidence = [e for k, _ in ranked for e in prof.evidence.get(k, [])]
        report.retrieval = self.retriever.retrieve(
            prof.category, evidence, alt_categories=[k for k, _ in ranked[1:3]], agentic=self.agentic)
        report.timings["retrieve_s"] = time.perf_counter() - t1

        return report


def featurize_csv(path) -> pd.DataFrame:
    """Featurize a user-supplied CSV with SOFI columns (``device`` column optional)."""
    import io

    from . import data

    text = open(path).read().replace("\r\n", "\n").replace("\r", "\n")
    df = pd.read_csv(io.StringIO(text)).rename(columns={"SNMP_Availability": "SNMP_availability"})
    if "device" not in df:
        df.insert(0, "device", "switch")
    if "class" in df:
        df["label"] = (df["class"] == "F").astype(int)
    for col in data.SPEED_COLS + data.TYPE_COLS:
        if col not in df:
            df[col] = -1
    return data.add_features(data.clean(df))


__all__ = ["DiagnosisAssistant", "DiagnosisReport", "featurize_csv", "config"]
