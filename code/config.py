"""
Configuration module for the Support Triage Agent.
All paths, constants, and API settings are centralized here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SUPPORT_TICKETS_DIR = PROJECT_ROOT / "support_tickets"

CORPUS_DIRS = {
    "hackerrank": DATA_DIR / "hackerrank",
    "claude": DATA_DIR / "claude",
    "visa": DATA_DIR / "visa",
}

INPUT_CSV = SUPPORT_TICKETS_DIR / "support_tickets.csv"
SAMPLE_CSV = SUPPORT_TICKETS_DIR / "sample_support_tickets.csv"
OUTPUT_CSV = SUPPORT_TICKETS_DIR / "output.csv"

# ─── LLM Settings ────────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_TEMPERATURE = 0.1  # Low temperature for deterministic output
GEMINI_MAX_TOKENS = 1024

# ─── Retrieval Settings ──────────────────────────────────────────────────────
TFIDF_MAX_FEATURES = 10000
TFIDF_NGRAM_RANGE = (1, 2)  # Unigrams and bigrams
RETRIEVAL_TOP_K = 5
CHUNK_SIZE = 500  # Words per chunk
CHUNK_OVERLAP = 50  # Words overlap between chunks
CONFIDENCE_THRESHOLD = 0.15  # Below this TF-IDF score → escalate

# ─── Valid Output Labels ─────────────────────────────────────────────────────
VALID_STATUSES = ["replied", "escalated"]

VALID_REQUEST_TYPES = ["product_issue", "feature_request", "bug", "invalid"]

# Fixed product area labels — consistent across all tickets
VALID_PRODUCT_AREAS = [
    # HackerRank areas
    "screen",               # Tests, assessments, screening
    "interviews",           # CodePair, live interviews
    "library",              # Question library
    "community",            # HackerRank Community (practice, contests, certs)
    "engage",               # Hiring events
    "integrations",         # ATS, SSO, scheduling integrations
    "settings",             # Account settings, roles, teams, API
    "skillup",              # SkillUp learning platform
    "chakra",               # Chakra product
    # Claude areas
    "account_management",   # Account, login, deletion, billing
    "conversation_management",  # Chats, memory, search
    "features",             # Artifacts, projects, skills, web search
    "api",                  # Claude API, Console, Bedrock
    "claude_code",          # Claude Code product
    "claude_desktop",       # Desktop app
    "claude_mobile",        # iOS/Android apps
    "claude_education",     # Education plans, LTI
    "privacy",              # Privacy, legal, data handling
    "plans_billing",        # Pro/Max/Team/Enterprise plans & billing
    "safeguards",           # Safety, security, bug bounty, content policy
    "connectors",           # MCP, integrations, Slack, GitHub
    "identity_management",  # SSO, SCIM, JIT
    # Visa areas
    "general_support",      # Lost/stolen cards, ATM, general FAQs
    "travel_support",       # Travel services, GCAS
    "fraud",                # Fraud prevention, identity theft
    "dispute_resolution",   # Chargebacks, disputes
    "data_security",        # Data breach response
    "regulations_fees",     # Rules, interchange, surcharging
    "checkout_fees",        # Checkout/merchant fees
    # Cross-domain
    "general",              # Generic or unclassifiable
    "out_of_scope",         # Not related to any supported product
]

# ─── High-Risk Keywords for Escalation ────────────────────────────────────────
HIGH_RISK_KEYWORDS = [
    # Fraud & Security
    "fraud", "unauthorized", "hacked", "identity theft", "identity stolen",
    "stolen", "compromised", "phishing", "scam",
    # Financial
    "refund", "charged twice", "payment failed", "payment issue",
    "billing dispute", "chargeback", "money back",
    # Account Access (requiring admin action)
    "account locked", "locked out", "restore access", "restore my access",
    "can't login", "cannot login", "can't log in", "cannot log in",
    # Subscription Management (requires system action)
    "pause subscription", "pause our subscription", "cancel subscription",
    "cancel my subscription",
    # Score Disputes
    "increase my score", "change my score", "review my answers",
    "graded unfairly", "graded me unfairly",
    # Security Vulnerabilities
    "security vulnerability", "vulnerability", "bug bounty",
    # Sensitive PII
    "order id", "order_id", "cs_live",
]

# Keywords that suggest a ticket should be replied as invalid
INVALID_KEYWORDS = [
    "delete all files", "rm -rf", "drop table", "sql injection",
    "iron man", "actor", "movie", "recipe", "weather",
]

# Keywords for company detection when company is None
COMPANY_KEYWORDS = {
    "hackerrank": [
        "hackerrank", "hacker rank", "codepair", "assessment",
        "test", "candidate", "interviewer", "screen", "coding challenge",
        "certification", "prep kit", "mock interview",
    ],
    "claude": [
        "claude", "anthropic", "artifact", "prompt", "conversation",
        "bedrock", "mcp", "claude code", "claude desktop",
    ],
    "visa": [
        "visa", "card", "merchant", "cardholder", "transaction",
        "atm", "traveller", "cheque", "3-d secure", "mastercard",
    ],
}
