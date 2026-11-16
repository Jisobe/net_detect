"""Command-line entry point: ``netdiag <command>`` (or ``python -m netdiag.cli``)."""

from __future__ import annotations

import argparse
import json
import sys

from . import config


def cmd_download_data(args):
    from . import data

    data.download_sofi(force=args.force)
    print(f"SOFI CSVs in {config.RAW_DIR}")


def cmd_train(args):
    from . import classifier

    detector, results = classifier.train_and_compare(run_cv=not args.no_cv)
    detector.save()
    classifier.write_report(results)
    print((config.REPORTS_DIR / "classifier_metrics.md").read_text())
    print(f"Saved {detector.name} detector (threshold {detector.threshold:.3f}) to {config.MODEL_PATH}")


def cmd_fetch_kb(args):
    from .kb import fetch

    manifest = fetch.fetch_all(force=args.force)
    print(f"{len(manifest)} documents in {config.KB_RAW_DIR}")


def cmd_build_kb(args):
    from .kb import ingest

    n = ingest.build_index()
    print(f"Indexed {n} chunks into {config.VECTOR_DB_DIR}")


def _select_row(args):
    import pandas as pd

    from . import data
    from .pipeline import featurize_csv

    if args.csv:
        df = featurize_csv(args.csv)
        if args.timestamp:
            df = df[df["timestamp"] == args.timestamp]
        return df.iloc[-1]
    df = data.load_dataset()
    if args.device:
        df = df[df["device"] == args.device]
    if args.timestamp:
        df = df[df["timestamp"] == args.timestamp]
    elif args.example:
        df = df[data.assign_splits(df) == "test"]
        df = df[df["label"] == (1 if args.example == "fault" else 0)]
        df = df.sample(1, random_state=args.seed)
    if df.empty:
        sys.exit("No matching sample.")
    return df.iloc[0] if isinstance(df, pd.DataFrame) else df


def cmd_diagnose(args):
    from .pipeline import DiagnosisAssistant

    row = _select_row(args)
    assistant = DiagnosisAssistant(args.backend, agentic=not args.no_agentic)
    report = assistant.diagnose(row, always_explain=args.always_explain)
    print(report.to_markdown())
    if not report.is_fault and not args.always_explain:
        print("\nNo fault detected. Use --always-explain to retrieve guidance anyway.")
    print("\nTimings: " + ", ".join(f"{k}={v:.2f}s" for k, v in report.timings.items()))


def cmd_eval_retrieval(args):
    from .evaluation.retrieval_eval import evaluate_retrieval

    evaluate_retrieval(k=args.k)
    print((config.REPORTS_DIR / "retrieval_metrics.md").read_text())


def cmd_all(args):
    cmd_download_data(argparse.Namespace(force=False))
    cmd_train(argparse.Namespace(no_cv=False))
    cmd_fetch_kb(argparse.Namespace(force=False))
    cmd_build_kb(args)
    cmd_eval_retrieval(argparse.Namespace(k=10))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="netdiag", description="ML + RAG wired-network fault diagnosis assistant")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download-data", help="download the SOFI dataset")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_download_data)

    p = sub.add_parser("train", help="train/compare GBDT, random forest and MLP fault detectors")
    p.add_argument("--no-cv", action="store_true", help="skip grouped cross-validation")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("fetch-kb", help="download troubleshooting documents for the knowledge base")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_fetch_kb)

    sub.add_parser("build-kb", help="chunk, embed and index the knowledge base").set_defaults(func=cmd_build_kb)

    p = sub.add_parser("diagnose", help="diagnose one sample")
    p.add_argument("--example", choices=["fault", "normal"], help="random test-split sample")
    p.add_argument("--device", choices=list(config.SOFI_FILES))
    p.add_argument("--timestamp", type=int, help="unix timestamp of the SOFI sample")
    p.add_argument("--csv", help="your own CSV with SOFI columns; diagnoses the last row")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", default="auto", choices=["auto", "anthropic", "ollama", "extractive"])
    p.add_argument("--always-explain", action="store_true", help="run RAG even if no fault is detected")
    p.add_argument("--no-agentic", action="store_true", help="disable query reformulation")
    p.add_argument("--groundedness", action="store_true", help="score groundedness of the output")
    p.set_defaults(func=cmd_diagnose)

    p = sub.add_parser("eval-retrieval", help="Precision@k / Recall@k of the retriever")
    p.add_argument("--k", type=int, default=10)
    p.set_defaults(func=cmd_eval_retrieval)

    p = sub.add_parser("eval-rag", help="classifier-only vs classifier+RAG comparison with groundedness")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--backend", default="auto", choices=["auto", "anthropic", "ollama", "extractive"])
    p.set_defaults(func=cmd_eval_rag)

    p = sub.add_parser("all", help="run the full pipeline end to end")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--backend", default="auto", choices=["auto", "anthropic", "ollama", "extractive"])
    p.set_defaults(func=cmd_all)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
