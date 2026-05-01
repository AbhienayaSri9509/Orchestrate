"""
fix_output.py — One-shot cleanup script.
Strips HTML tags & entities from the 'response' column of output.csv.
Run once from the repo root:  python fix_output.py
"""
import re
import csv
from pathlib import Path

INPUT  = Path("support_tickets/output.csv")
OUTPUT = Path("support_tickets/output.csv")


def strip_html(text: str) -> str:
    """Remove HTML tags and decode common HTML entities."""
    text = text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    text = text.replace("&nbsp;", " ").replace("&#39;", "'").replace("&quot;", '"')
    text = re.sub(r"<[^>]+>", " ", text)           # strip tags
    text = re.sub(r" {2,}", " ", text)              # collapse spaces
    text = re.sub(r"\n{3,}", "\n\n", text)          # collapse blank lines
    return text.strip()


def main():
    rows = []
    with open(INPUT, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            row["response"] = strip_html(row["response"])
            rows.append(row)

    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done — {len(rows)} rows cleaned and written to {OUTPUT}")


if __name__ == "__main__":
    main()
