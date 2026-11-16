"""Central paths and tunable settings for the network diagnosis assistant."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("NETDIAG_ROOT", Path(__file__).resolve().parents[1]))

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
KB_DIR = DATA_DIR / "kb"
KB_RAW_DIR = KB_DIR / "raw"          # documents downloaded by `netdiag fetch-kb`
KB_LOCAL_DIR = KB_DIR / "local"      # user-supplied PDFs / HTML / TXT (e.g. vendor guides)
KB_CHUNKS_PATH = KB_DIR / "chunks.jsonl"
VECTOR_DB_DIR = KB_DIR / "chroma"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "fault_detector.joblib"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

# --- SOFI dataset (Mendeley Data, doi:10.17632/tc6ysmh5j8.2, CC BY 4.0) ---
SOFI_FILES = {
    "CoreSwitch-I": {
        "filename": "SOFI_CoreSwitch-I.csv",
        "url": "https://data.mendeley.com/public-files/datasets/tc6ysmh5j8/files/"
               "8c35a40c-e552-4a8e-84ae-498d4f0aabb6/file_downloaded",
        "sha256": "e023440004e66fdf2f26737d73cce4f943f4a3435bdebf3676f3d965152a0b49",
    },
    "CoreSwitch-II": {
        "filename": "SOFI_CoreSwitch-II.csv",
        "url": "https://data.mendeley.com/public-files/datasets/tc6ysmh5j8/files/"
               "657b5af9-7089-4015-a4db-58eafce22de6/file_downloaded",
        "sha256": "5d471944cc6a93e7095f90be0ba64aad926f7160b50533016677edef6e81a048",
    },
}

SAMPLE_INTERVAL_S = 180       # SOFI polls every 3 minutes
BASELINE_WINDOW = 10          # samples (30 min) used for the rolling "normal" baseline
RANDOM_SEED = 42

# --- Retrieval ---
EMBEDDING_MODEL = os.environ.get("NETDIAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE_WORDS = 180
CHUNK_OVERLAP_WORDS = 40
TOP_K = 5
# Below this best-hit cosine similarity the agentic step reformulates the query.
RETRIEVAL_CONFIDENCE_THRESHOLD = 0.45
