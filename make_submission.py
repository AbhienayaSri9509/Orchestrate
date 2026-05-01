"""
make_submission.py
Run this from the repo root to prepare all 3 submission files.

Usage:
    python make_submission.py
"""
import zipfile
import shutil
from pathlib import Path

REPO = Path(__file__).parent
CODE = REPO / "code"
ZIP_OUT = REPO / "code_submission.zip"
OUTPUT_CSV = REPO / "support_tickets" / "output.csv"
LOG_TXT = Path.home() / "hackerrank_orchestrate" / "log.txt"

EXCLUDE_DIRS = {
    "__pycache__", ".git", "node_modules",
    "dist", "build", ".venv", "venv",
    ".mypy_cache", ".ruff_cache", ".pytest_cache",
}
EXCLUDE_EXTS = {".pyc", ".pyo", ".db", ".sqlite"}


def make_zip():
    """Zip code/ directory, excluding unwanted paths."""
    if ZIP_OUT.exists():
        ZIP_OUT.unlink()

    with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in CODE.rglob("*"):
            # Skip excluded directory names anywhere in path
            if any(part in EXCLUDE_DIRS for part in f.parts):
                continue
            if f.suffix in EXCLUDE_EXTS:
                continue
            if f.is_file():
                arcname = "code/" + f.relative_to(CODE).as_posix()
                zf.write(f, arcname)
                print(f"  + {arcname}")

    size_kb = round(ZIP_OUT.stat().st_size / 1024, 1)
    print(f"\n✅ ZIP created: {ZIP_OUT}  ({size_kb} KB)")
    return ZIP_OUT


def check_csv():
    if OUTPUT_CSV.exists():
        lines = OUTPUT_CSV.read_text(encoding="utf-8").strip().splitlines()
        print(f"✅ Predictions CSV: {OUTPUT_CSV}  ({len(lines) - 1} rows)")
    else:
        print(f"❌ Missing: {OUTPUT_CSV}")
    return OUTPUT_CSV


def check_log():
    if LOG_TXT.exists():
        size_kb = round(LOG_TXT.stat().st_size / 1024, 1)
        print(f"✅ Chat transcript: {LOG_TXT}  ({size_kb} KB)")
    else:
        print(f"❌ Missing: {LOG_TXT}")
    return LOG_TXT


def main():
    print("=" * 60)
    print("  HackerRank Orchestrate — Submission File Prep")
    print("=" * 60)
    print()

    print("[1/3] Creating code zip...")
    zip_path = make_zip()
    print()

    print("[2/3] Checking predictions CSV...")
    csv_path = check_csv()
    print()

    print("[3/3] Checking chat transcript...")
    log_path = check_log()
    print()

    print("=" * 60)
    print("  SUBMISSION FILES READY")
    print("=" * 60)
    print(f"  1. Code zip      : {zip_path}")
    print(f"  2. Predictions   : {csv_path}")
    print(f"  3. Chat transcript: {log_path}")
    print()
    print("Upload all three at:")
    print("  https://www.hackerrank.com/contests/hackerrank-orchestrate-may26/challenges/support-agent/submission")
    print()


if __name__ == "__main__":
    main()
