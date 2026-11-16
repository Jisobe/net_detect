# ML + RAG Assistant for Wired Network Fault Diagnosis

1. **Fault detector:** a supervised classifier trained on the SOFI dataset's symptom features. Gradient-boosted trees, a random forest and a neural network (MLP) are compared.
2. **RAG diagnosis:** the detected sample is profiled into a fault category and described in plain language. That description is the query against a vector index (sentence-transformers + ChromaDB) of public troubleshooting documentation. An LLM then writes a root-cause explanation and remediation plan that cites the retrieved passages. When retrieval confidence is low, an agentic step reformulates the query and retrieves again.

```txt
SOFI telemetry ─> clean + causal features ─> fault detector (RF / GBDT / MLP)
                                              │ fault?
                                              v
                         symptom profiler ─▶ category + NL description
                                              │
           RFCs / Wikipedia / Linux / VyOS ─▶ Chroma (MiniLM embeddings) ─▶ top-k passages
                                              │  (low confidence → query reformulation)
                                              v
                     LLM (Claude / local Mistral via Ollama / extractive fallback)
                                              v
                    cited root cause + remediation + groundedness score
```

## Quick start

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
source .venv/bin/activate

netdiag all --backend auto           # download data, train, build KB, run all evaluations
# or step by step:
netdiag download-data                # SOFI CSVs from Mendeley Data (SHA-256 verified)
netdiag train                        # compares GBDT / RF / MLP, saves the best detector
netdiag fetch-kb && netdiag build-kb # download docs, chunk, embed, index
netdiag diagnose --example fault --groundedness
netdiag diagnose --device CoreSwitch-II --timestamp 1568676060
netdiag diagnose --csv my_poll.csv   # your own data with SOFI columns (last row is diagnosed)
netdiag eval-retrieval
netdiag eval-rag --n 20 --backend anthropic
pytest
```

## Layout

```txt
src/netdiag/
  config.py           paths, dataset URLs/hashes, retrieval + model settings
  data.py             SOFI loading, sentinel cleanup, causal temporal features, grouped splits
  classifier.py       GBDT vs RF vs MLP, threshold tuning, grouped CV, metrics report
  taxonomy.py         fault categories (mapped to the proposal's fault list) + relevance patterns
  symptoms.py         symptom profiler: category evidence scores + NL description
  kb/sources.py       knowledge-base source list
  kb/fetch.py         downloader (RFC Editor, Wikipedia API, kernel.org, VyOS)
  kb/ingest.py        RFC/Wikipedia/HTML/PDF parsing, chunking, Chroma indexing
  retrieval.py        embeddings, vector store, agentic query reformulation
  generation.py       prompt + Anthropic / Ollama / extractive backends
  pipeline.py         DiagnosisAssistant (end-to-end) and report rendering
  evaluation/         retrieval P@k/R@k, groundedness, classifier vs classifier+RAG
  cli.py              `netdiag` command
artifacts/reports/    generated metrics (markdown + JSON)
data/kb/local/        drop your own vendor PDFs/HTML here, then `netdiag build-kb`
```

## Data findings that shaped the design

The [SOFI dataset](https://data.mendeley.com/datasets/tc6ysmh5j8/2) (CC BY 4.0) has two CSVs, one per core switch. Each has 12,971 samples polled every 180 s from April to September 2019.

- **The labels are binary.** `class` is only `NE` (no error) or `F` (fault): 196 fault samples per switch (1.5%), in 38 episodes. Both switches share the same episodes and timestamps. The proposal's fault categories (link down, high utilization, faulty cabling, …) are therefore not labelled. Stage 1 is a fault **detector**, and the category is inferred by a transparent rule-based **symptom profiler** (`symptoms.py`, `taxonomy.py`). Treat the category as a hypothesis for the engineer, not a classifier output.
- **Faults mostly look like traffic collapse with links up.** Bits sent and received drop by more than 95% while every interface stays operationally up, sometimes with bursts of input errors. Most faults are profiled as `traffic_loss`, the rest as `interface_errors` or `unclassified`.
- **`-1`/`-2` values are sentinels for "not collected"**, as is `IN_Operational_status = 0`. They are converted to NaN plus missing-value flags. `ICMP_loss` values above 100% are discarded. `timestamp`, `range` and absolute `Device_uptime` are not model inputs because they are identifier-like or spuriously correlated.
- **Features are causal.** Each traffic metric is compared with its rolling median over the previous 30 minutes, within contiguous polling segments, plus forwarding-balance and short-memory features. A test checks that the detector never sees future samples, so it can run online.
- **Splits are leakage-safe.** The proposal asks for stratified train/val/test splits. A random row split would put adjacent samples of one episode, and the same episode seen from the other switch, on both sides. Splits are therefore stratified over whole one-day blocks shared by both switches (`StratifiedGroupKFold`).

## Results

All numbers below come from `artifacts/reports/`, regenerated by the commands above.

### Stage 1: fault detector

| model | test accuracy | fault precision | fault recall | fault F1 | test PR-AUC | CV PR-AUC (5-fold, grouped) | CV ROC-AUC |
|---|---|---|---|---|---|---|---|
| GBDT (HistGradientBoosting) | 0.978 | 0.46 | 0.32 | 0.38 | 0.40 | 0.42 ± 0.15 | 0.91 ± 0.06 |
| Random forest (**selected**) | 0.980 | 0.60 | 0.24 | 0.34 | 0.35 | **0.50 ± 0.15** | **0.95 ± 0.03** |
| MLP | 0.973 | 0.39 | 0.47 | 0.43 | 0.42 | 0.39 ± 0.15 | 0.93 ± 0.04 |

- **Accuracy target:** every model clears the proposal's 85% target, but that number is not informative here. Predicting "no fault" for every sample already scores 98.5%.
- **Metrics that matter:** fault-class F1 and PR-AUC.
- **Model selection:** by grouped cross-validation, since the test split holds only about 8 episodes. The random forest is clearly best on CV, while the MLP happens to do best on this particular test split.
- **Why fault recall is limited:**
  - The first sample of almost every labelled episode looks normal: median traffic change is +12%, so the label starts before the symptom does.
  - Comparable traffic drops also occur in about 6% of normal samples, because traffic on this switch is low-volume and bursty.
- **Episode-level detection** (at least one sample flagged per episode) is 38–69% on the test split, depending on the model.

## Limitations and next steps

- **No ground-truth categories.** Fault categories are inferred by rules, not learned. A dataset with typed fault labels would allow a real multi-class classifier; `taxonomy.py` and `symptoms.py` were written so one can be swapped in.
- **Retrieval relevance is a keyword proxy.** For the final report, hand-label a sample of query–chunk pairs (start from `qrels.json`) or use an LLM judge.
- **The diagnosis-time claim needs people.** The proposal's "faster than a junior engineer" claim needs a timed user study. Suggested protocol: give engineers the same SOFI episodes with (a) the raw dashboard, (b) the detector verdict, (c) the full RAG report. Measure time to a correct root-cause hypothesis and rate the remediation quality.
- **Vendor guides are not fetched automatically.** Cisco TAC notes block scripted downloads. Save them from a browser into `data/kb/local/` and rebuild the index.
