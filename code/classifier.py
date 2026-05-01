"""
Ticket Classifier

Classifies tickets by request_type and product_area.
Primary: uses Gemini LLM with structured prompt.
Fallback: keyword-based rules if LLM is unavailable or fails.
"""

import re
import json
from typing import Dict, Optional, List, Tuple

from config import (
    VALID_REQUEST_TYPES, VALID_PRODUCT_AREAS, COMPANY_KEYWORDS,
    GOOGLE_API_KEY, GEMINI_MODEL, GEMINI_TEMPERATURE,
)

# Try importing Gemini — graceful fallback if unavailable
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = bool(GOOGLE_API_KEY)
    if GEMINI_AVAILABLE:
        genai.configure(api_key=GOOGLE_API_KEY)
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None


def detect_company(issue: str, subject: str, company: str) -> str:
    """
    Detect or confirm the company from ticket content.
    If company is None/empty, infer from keywords.
    """
    if company and company.strip().lower() not in ("none", ""):
        return company.strip()

    combined = f"{issue} {subject}".lower()

    scores = {}
    for comp, keywords in COMPANY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in combined)
        if score > 0:
            scores[comp] = score

    if scores:
        best = max(scores, key=scores.get)
        return best.capitalize() if best != "hackerrank" else "HackerRank"

    return "None"


def _keyword_classify_request_type(issue: str) -> str:
    """Fallback keyword-based request type classification."""
    issue_lower = issue.lower()

    # Bug indicators
    bug_keywords = [
        "not working", "error", "crash", "broken", "down",
        "failing", "failed", "can't", "cannot", "unable",
        "blocker", "blocked", "issue", "problem", "bug",
        "stopped", "doesn't work", "won't work",
    ]
    bug_score = sum(1 for kw in bug_keywords if kw in issue_lower)

    # Feature request indicators
    feature_keywords = [
        "feature request", "add feature", "would be nice",
        "can you add", "suggestion", "enhance", "improvement",
        "extend", "can we have", "i wish", "it would help",
    ]
    feature_score = sum(1 for kw in feature_keywords if kw in issue_lower)

    # Product issue indicators (how-to, help, guidance)
    product_keywords = [
        "how to", "how do", "how can", "help me", "guide",
        "steps", "instructions", "what is", "where can",
        "please", "i need", "i want", "looking for",
        "set up", "setup", "configure", "create",
    ]
    product_score = sum(1 for kw in product_keywords if kw in issue_lower)

    # Score-based decision
    if feature_score > 0 and feature_score >= bug_score:
        return "feature_request"
    elif bug_score >= 3:
        return "bug"
    elif bug_score > product_score and bug_score >= 2:
        return "bug"
    elif product_score > 0:
        return "product_issue"
    elif bug_score > 0:
        return "product_issue"  # Single bug keyword → likely a product issue
    else:
        return "product_issue"  # Default


def _keyword_classify_product_area(issue: str, subject: str, company: str) -> str:
    """Fallback keyword-based product area classification."""
    combined = f"{issue} {subject}".lower()
    company_lower = company.lower() if company else ""

    # ── HackerRank Areas ──
    if company_lower == "hackerrank":
        if any(kw in combined for kw in ["test", "assessment", "candidate", "screen", "invite", "variant", "time accommodation"]):
            return "screen"
        if any(kw in combined for kw in ["interview", "codepair", "lobby", "inactivity"]):
            return "interviews"
        if any(kw in combined for kw in ["question", "library", "coding question"]):
            return "library"
        if any(kw in combined for kw in ["community", "practice", "certification", "certificate", "contest", "mock interview", "prep kit", "challenge"]):
            return "community"
        if any(kw in combined for kw in ["subscription", "billing", "payment", "money", "refund"]):
            return "community"  # HackerRank billing is under community
        if any(kw in combined for kw in ["event", "engage", "hackathon", "leaderboard"]):
            return "engage"
        if any(kw in combined for kw in ["integration", "ats", "sso", "single sign"]):
            return "integrations"
        if any(kw in combined for kw in ["role", "team", "admin", "user", "remove", "setting", "api", "infosec"]):
            return "settings"
        if any(kw in combined for kw in ["skillup", "learning"]):
            return "skillup"
        if any(kw in combined for kw in ["resume", "apply tab"]):
            return "community"
        return "screen"  # Default for HackerRank

    # ── Claude Areas ──
    if company_lower == "claude":
        if any(kw in combined for kw in ["account", "login", "delete", "email", "session"]):
            return "account_management"
        if any(kw in combined for kw in ["conversation", "chat", "delete", "rename", "memory"]):
            return "conversation_management"
        if any(kw in combined for kw in ["api", "console", "bedrock", "aws", "429", "rate limit"]):
            return "api"
        if any(kw in combined for kw in ["claude code", "code review", "security review"]):
            return "claude_code"
        if any(kw in combined for kw in ["desktop", "mcp", "extension"]):
            return "claude_desktop"
        if any(kw in combined for kw in ["mobile", "ios", "android", "app"]):
            return "claude_mobile"
        if any(kw in combined for kw in ["education", "university", "lti", "professor", "student"]):
            return "claude_education"
        if any(kw in combined for kw in ["privacy", "data", "crawl", "training", "gdpr"]):
            return "privacy"
        if any(kw in combined for kw in ["billing", "subscription", "plan", "pro", "max", "team", "enterprise", "seat", "workspace"]):
            return "plans_billing"
        if any(kw in combined for kw in ["safety", "bug bounty", "vulnerability", "safeguard", "abuse"]):
            return "safeguards"
        if any(kw in combined for kw in ["connector", "slack", "github", "integration"]):
            return "connectors"
        if any(kw in combined for kw in ["sso", "scim", "identity"]):
            return "identity_management"
        if any(kw in combined for kw in ["not working", "error", "failing", "stopped", "not responding"]):
            return "account_management"  # General troubleshooting
        return "features"  # Default for Claude

    # ── Visa Areas ──
    if company_lower == "visa":
        if any(kw in combined for kw in ["fraud", "identity theft", "identity stolen", "scam", "phishing"]):
            return "fraud"
        if any(kw in combined for kw in ["dispute", "chargeback", "charged", "refund"]):
            return "dispute_resolution"
        if any(kw in combined for kw in ["travel", "abroad", "overseas", "atm", "currency", "gcas"]):
            return "travel_support"
        if any(kw in combined for kw in ["lost", "stolen", "damaged", "blocked", "declined", "emergency", "cash"]):
            return "general_support"
        if any(kw in combined for kw in ["data", "breach", "compromise", "security"]):
            return "data_security"
        if any(kw in combined for kw in ["rule", "regulation", "fee", "interchange", "surcharge", "minimum", "maximum"]):
            return "regulations_fees"
        if any(kw in combined for kw in ["merchant", "checkout"]):
            return "checkout_fees"
        return "general_support"  # Default for Visa

    # ── Unknown Company ──
    return "general"


def _llm_classify(
    issue: str,
    subject: str,
    company: str,
    retrieved_context: str,
) -> Optional[Dict]:
    """
    Use Gemini to classify the ticket. Returns dict with request_type and product_area.
    Returns None if LLM fails.
    """
    if not GEMINI_AVAILABLE:
        return None

    valid_areas_str = ", ".join(VALID_PRODUCT_AREAS)

    prompt = f"""You are a support ticket classifier. Classify this ticket ONLY based on the provided context.

TICKET:
- Issue: {issue}
- Subject: {subject}
- Company: {company}

RETRIEVED SUPPORT DOCS (for reference):
{retrieved_context[:2000]}

INSTRUCTIONS:
1. Determine the request_type: one of [product_issue, feature_request, bug, invalid]
2. Determine the product_area: one of [{valid_areas_str}]
3. Determine if this should be escalated (true/false)

RULES:
- "product_issue" = user needs help with a product feature or process
- "bug" = something is broken, not working, or producing errors
- "feature_request" = user wants a new feature or enhancement
- "invalid" = off-topic, malicious, gratitude, or out of scope
- Escalate if: fraud, billing dispute, account access issues, security vulnerabilities, subscription changes, score disputes, or if no relevant documentation exists

Respond ONLY with valid JSON (no markdown, no explanation):
{{"request_type": "...", "product_area": "...", "should_escalate": true/false, "escalation_reason": "..."}}"""

    try:
        model = genai.GenerativeModel(
            GEMINI_MODEL,
            generation_config={
                "temperature": GEMINI_TEMPERATURE,
                "max_output_tokens": 256,
                "response_mime_type": "application/json",
            }
        )
        response = model.generate_content(prompt)
        result = json.loads(response.text)

        # Validate the output
        if result.get("request_type") not in VALID_REQUEST_TYPES:
            result["request_type"] = "product_issue"
        if result.get("product_area") not in VALID_PRODUCT_AREAS:
            result["product_area"] = _keyword_classify_product_area(issue, subject, company)

        return result

    except Exception as e:
        print(f"    [WARN] LLM classification failed: {e}")
        return None


def classify_ticket(
    issue: str,
    subject: str,
    company: str,
    retrieved_context: str = "",
) -> Dict:
    """
    Classify a ticket using LLM with keyword fallback.

    Returns:
        {
            "request_type": str,
            "product_area": str,
            "llm_escalate": bool,
            "llm_reason": str,
            "method": "llm" | "keyword"
        }
    """
    # Try LLM first
    llm_result = _llm_classify(issue, subject, company, retrieved_context)

    if llm_result:
        return {
            "request_type": llm_result.get("request_type", "product_issue"),
            "product_area": llm_result.get("product_area", "general"),
            "llm_escalate": llm_result.get("should_escalate", False),
            "llm_reason": llm_result.get("escalation_reason", ""),
            "method": "llm",
        }

    # Fallback to keyword-based classification
    return {
        "request_type": _keyword_classify_request_type(issue),
        "product_area": _keyword_classify_product_area(issue, subject, company),
        "llm_escalate": False,
        "llm_reason": "",
        "method": "keyword",
    }
