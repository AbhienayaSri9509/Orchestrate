"""
Multi-Domain Support Triage Agent — Main Pipeline

Terminal-based agent that processes support tickets from CSV,
classifies them, retrieves relevant corpus documentation,
and generates safe, grounded responses.

Usage:
    python main.py                          # Process support_tickets.csv
    python main.py --sample                 # Process sample_support_tickets.csv (for testing)
    python main.py --input path/to/input.csv --output path/to/output.csv

Architecture:
    1. Load & index corpus (TF-IDF)
    2. For each ticket:
       a. Detect/confirm company
       b. Retrieve relevant corpus chunks
       c. Run risk assessment (deterministic rules)
       d. Classify request type & product area (LLM + keyword fallback)
       e. Decide: reply vs escalate
       f. Generate corpus-grounded response
       g. Generate justification
    3. Write output CSV
"""

import sys
import csv
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    filename="log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)
# Also log to console (INFO+)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.WARNING)
logging.getLogger().addHandler(console_handler)
# ──────────────────────────────────────────────────────────────────────────────

# Add code/ to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import INPUT_CSV, SAMPLE_CSV, OUTPUT_CSV, GOOGLE_API_KEY
from corpus_loader import CorpusRetriever
from risk_detector import full_risk_assessment
from classifier import classify_ticket, detect_company
from response_generator import (
    generate_response, generate_justification, _format_retrieved_context,
)


def print_banner():
    """Print a startup banner."""
    print("=" * 70)
    print("  Multi-Domain Support Triage Agent")
    print("  HackerRank Orchestrate — May 2026")
    print("=" * 70)
    print()


def print_ticket_summary(idx: int, issue: str, company: str, status: str,
                         request_type: str, product_area: str, method: str):
    """Print a concise summary for each processed ticket."""
    # Truncate issue for display
    short_issue = issue[:80].replace("\n", " ").strip()
    if len(issue) > 80:
        short_issue += "..."

    status_icon = "✅" if status == "replied" else "⚠️"
    print(f"  [{idx:02d}] {status_icon} {status.upper():10s} | {request_type:16s} | {product_area:25s} | {method:7s} | {short_issue}")


def load_tickets(csv_path: Path) -> List[Dict]:
    """Load support tickets from CSV."""
    tickets = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tickets.append({
                "issue": row.get("Issue", row.get("issue", "")).strip(),
                "subject": row.get("Subject", row.get("subject", "")).strip(),
                "company": row.get("Company", row.get("company", "")).strip(),
            })
    return tickets


def process_ticket(
    ticket: Dict,
    retriever: CorpusRetriever,
) -> Dict:
    """
    Process a single support ticket through the full pipeline.

    Returns a dict with all output fields.
    """
    issue = ticket["issue"]
    subject = ticket["subject"]
    raw_company = ticket["company"]

    # Step 1: Detect/confirm company
    company = detect_company(issue, subject, raw_company)

    # Step 2: Retrieve relevant corpus chunks
    query = f"{issue} {subject}".strip()
    company_hint = company if company.lower() not in ("none", "") else None
    retrieved_chunks = retriever.retrieve(query, company_hint)
    top_score = retrieved_chunks[0][1] if retrieved_chunks else 0.0

    logger.info("Processing ticket | company=%s | subject=%s", company, subject[:60])
    logger.info("  Top retrieval score: %.3f", top_score)

    # Step 3: Run risk assessment (deterministic)
    risk = full_risk_assessment(issue, subject, top_score)

    # Step 4: Classify request type & product area
    context_str = _format_retrieved_context(retrieved_chunks[:3])
    classification = classify_ticket(issue, subject, company, context_str)

    request_type = classification["request_type"]
    product_area = classification["product_area"]
    method = classification["method"]

    # Override classification for known cases
    if risk["is_gratitude"]:
        request_type = "invalid"
        product_area = "general"
    elif risk["is_invalid"]:
        request_type = "invalid"
        if product_area == "general":
            product_area = "out_of_scope"

    # Step 5: Decide reply vs escalate
    should_escalate = (
        risk["should_escalate"]
        or classification.get("llm_escalate", False)
    )

    # Don't escalate gratitude or clearly invalid out-of-scope
    if risk["is_gratitude"]:
        should_escalate = False
    if risk["is_invalid"] and not risk["should_escalate"]:
        should_escalate = False
    # Vague tickets → reply with clarification (don't escalate)
    if risk.get("is_vague") and not risk["should_escalate"]:
        should_escalate = False

    status = "escalated" if should_escalate else "replied"

    # Step 6: Generate response
    response = generate_response(
        issue=issue,
        subject=subject,
        company=company,
        product_area=product_area,
        status=status,
        request_type=request_type,
        risk_assessment=risk,
        retrieved_chunks=retrieved_chunks,
    )

    # Step 7: Generate justification
    justification = generate_justification(
        status=status,
        request_type=request_type,
        product_area=product_area,
        risk_assessment=risk,
        classification_method=method,
        top_retrieval_score=top_score,
        retrieved_chunks=retrieved_chunks,
    )

    logger.info(
        "  Result | status=%s | request_type=%s | product_area=%s | method=%s",
        status, request_type, product_area, method,
    )

    return {
        "issue": issue,
        "subject": subject,
        "company": company,
        "response": response,
        "product_area": product_area,
        "status": status,
        "request_type": request_type,
        "justification": justification,
        # Extra metadata (not written to CSV)
        "_method": method,
        "_top_score": top_score,
        "_risk_reasons": risk.get("reasons", []),
    }


def write_output(results: List[Dict], output_path: Path):
    """Write results to output CSV."""
    fieldnames = [
        "issue", "subject", "company", "response",
        "product_area", "status", "request_type", "justification",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for result in results:
            row = {k: result[k] for k in fieldnames}
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Domain Support Triage Agent"
    )
    parser.add_argument(
        "--sample", action="store_true",
        help="Process sample_support_tickets.csv instead of support_tickets.csv"
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to output CSV file"
    )
    args = parser.parse_args()

    print_banner()

    # Determine input/output paths
    if args.input:
        input_path = Path(args.input)
    elif args.sample:
        input_path = SAMPLE_CSV
    else:
        input_path = INPUT_CSV

    output_path = Path(args.output) if args.output else OUTPUT_CSV

    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"  LLM:    {'Gemini (' + 'available' + ')' if GOOGLE_API_KEY else 'Not available (keyword fallback)'}")
    print()

    # Step 1: Load & index corpus
    print("[1/4] Loading and indexing corpus...")
    start_time = time.time()
    logger.info("Starting triage agent — input: %s", input_path)
    logger.info("Loading and indexing corpus...")
    retriever = CorpusRetriever()
    index_time = time.time() - start_time
    print(f"  Corpus indexed in {index_time:.1f}s")
    logger.info("Corpus indexed in %.1fs", index_time)
    print()

    # Step 2: Load tickets
    print("[2/4] Loading tickets...")
    tickets = load_tickets(input_path)
    print(f"  Loaded {len(tickets)} tickets")
    logger.info("Loaded %d tickets from %s", len(tickets), input_path)
    print()

    # Step 3: Process each ticket
    print("[3/4] Processing tickets...")
    print("-" * 130)
    print(f"  {'#':>4s}  {'Status':10s} | {'Request Type':16s} | {'Product Area':25s} | {'Method':7s} | Issue")
    print("-" * 130)

    results = []
    start_time = time.time()

    for idx, ticket in enumerate(tickets, 1):
        result = process_ticket(ticket, retriever)
        results.append(result)

        print_ticket_summary(
            idx, result["issue"], result["company"],
            result["status"], result["request_type"],
            result["product_area"], result["_method"],
        )

    process_time = time.time() - start_time
    print("-" * 130)
    print()

    # Step 4: Write output
    print("[4/4] Writing output...")
    write_output(results, output_path)
    print(f"  Output written to: {output_path}")
    print()

    # Summary stats
    replied_count = sum(1 for r in results if r["status"] == "replied")
    escalated_count = sum(1 for r in results if r["status"] == "escalated")
    llm_count = sum(1 for r in results if r["_method"] == "llm")
    keyword_count = sum(1 for r in results if r["_method"] == "keyword")

    print("=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"  Total tickets:  {len(results)}")
    print(f"  Replied:        {replied_count}")
    print(f"  Escalated:      {escalated_count}")
    print(f"  LLM classified: {llm_count}")
    print(f"  Keyword classified: {keyword_count}")
    print(f"  Processing time: {process_time:.1f}s ({process_time/len(results):.1f}s/ticket)")
    print("=" * 70)
    print()
    print("Done! ✅")
    logger.info(
        "Processing complete | total=%d | replied=%d | escalated=%d | time=%.2fs",
        len(results), replied_count, escalated_count, process_time,
    )


if __name__ == "__main__":
    main()
