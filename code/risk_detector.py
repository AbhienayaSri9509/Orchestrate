"""
Risk Detector & Escalation Logic

Provides deterministic, keyword-based risk assessment for support tickets.
This is the FIRST line of defense — runs BEFORE any LLM classification.

High-risk tickets are always escalated, regardless of LLM output.
"""

import re
from typing import Dict, Tuple

from config import HIGH_RISK_KEYWORDS, INVALID_KEYWORDS, CONFIDENCE_THRESHOLD


def detect_high_risk(issue: str, subject: str = "") -> Tuple[bool, str]:
    """
    Check if a ticket contains high-risk content that requires immediate escalation.

    Returns:
        (should_escalate, reason)
    """
    combined = f"{issue} {subject}".lower().strip()

    # Check each high-risk keyword
    for keyword in HIGH_RISK_KEYWORDS:
        if keyword in combined:
            return True, f"High-risk keyword detected: '{keyword}'. Requires human review."

    return False, ""


def detect_invalid(issue: str, subject: str = "") -> Tuple[bool, str]:
    """
    Check if a ticket is clearly invalid (malicious, out-of-scope, or nonsensical).

    Returns:
        (is_invalid, reason)
    """
    combined = f"{issue} {subject}".lower().strip()

    # Check for malicious/prompt-injection patterns
    prompt_injection_patterns = [
        r"ignore (all |previous |above )?instructions",
        r"forget (all |previous |above )?instructions",
        r"you are now",
        r"act as",
        r"pretend to be",
        r"reveal (your|the) (system|internal|secret)",
        r"show (me |)(your |the )(system |)prompt",
        r"display (all |)(internal|system|secret)",
        r"affiche.*règles internes",  # French prompt injection
        r"logique exacte",            # French: "exact logic"
    ]
    for pattern in prompt_injection_patterns:
        if re.search(pattern, combined):
            return True, f"Potential prompt injection detected."

    # Check for clearly invalid/out-of-scope content
    for keyword in INVALID_KEYWORDS:
        if keyword in combined:
            return True, f"Out-of-scope request detected: '{keyword}'."

    # Check if the issue is too short/empty to be meaningful
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', issue).strip()
    if len(cleaned) < 10:
        return False, ""  # Short but not necessarily invalid — might need escalation

    return False, ""


def detect_gratitude(issue: str) -> bool:
    """Check if the ticket is just a thank-you message."""
    gratitude_patterns = [
        r"^thank(s| you)",
        r"^thanks?\s*(for|!|\.|\s)*$",
        r"^happy to help",
        r"^great\s*,?\s*thanks",
        r"^ok\s*,?\s*thanks",
        r"^appreciated",
    ]
    cleaned = issue.lower().strip()
    for pattern in gratitude_patterns:
        if re.search(pattern, cleaned):
            return True
    return False


def assess_confidence(top_retrieval_score: float) -> Tuple[bool, str]:
    """
    Assess if the retrieval confidence is too low for a safe response.

    Returns:
        (should_escalate, reason)
    """
    if top_retrieval_score < CONFIDENCE_THRESHOLD:
        return True, (
            f"Low retrieval confidence (score={top_retrieval_score:.3f}, "
            f"threshold={CONFIDENCE_THRESHOLD}). No relevant corpus documentation found."
        )
    return False, ""


def detect_multi_request(issue: str) -> bool:
    """
    Check if a ticket contains multiple distinct requests.
    Multi-request tickets may need more careful handling.
    """
    # Count distinct request patterns
    request_patterns = [
        r"(?:please|can you|could you|i need|i want|how (?:do|can|to))\s",
    ]
    count = 0
    for pattern in request_patterns:
        matches = re.findall(pattern, issue.lower())
        count += len(matches)

    return count >= 3  # 3+ distinct requests → flag as multi-request


def detect_vague_ticket(issue: str, subject: str = "") -> bool:
    """
    Detect if a ticket is too vague to meaningfully process.
    Vague tickets should get a clarification response instead of escalation.

    Examples: "not working help", "it's broken", "help needed"
    """
    combined = f"{issue} {subject}".lower().strip()
    # Remove punctuation for word counting
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', issue).strip()
    words = cleaned.split()

    # Very short tickets with generic complaint words
    if len(words) <= 6:
        vague_patterns = [
            r"not working",
            r"help$",
            r"^help",
            r"broken",
            r"fix it",
            r"doesn'?t work",
            r"won'?t work",
            r"it'?s down",
        ]
        for pattern in vague_patterns:
            if re.search(pattern, combined):
                return True

    return False


def detect_requires_account_action(issue: str) -> Tuple[bool, str]:
    """
    Detect if a ticket requires an action that can only be performed
    by accessing the user's account (which the agent cannot do).
    """
    account_action_patterns = [
        (r"delete my account", "Account deletion requires human verification."),
        (r"remove (?:me|my|an?) (?:from|account)", "Account/user removal requires admin action."),
        (r"pause (?:our|my|the) subscription", "Subscription changes require billing system access."),
        (r"cancel (?:our|my|the) subscription", "Subscription cancellation requires billing system access."),
        (r"update (?:my|the) (?:name|certificate)", "Profile/certificate updates require account access."),
        (r"reschedul", "Assessment rescheduling is handled by the hiring company, not HackerRank."),
        (r"fill(?:ing)? in the forms", "InfoSec/compliance form completion requires human support."),
    ]

    combined = issue.lower().strip()
    for pattern, reason in account_action_patterns:
        if re.search(pattern, combined):
            return True, reason

    return False, ""


def full_risk_assessment(
    issue: str,
    subject: str,
    top_retrieval_score: float,
) -> Dict:
    """
    Run all risk checks and return a comprehensive assessment.

    Returns dict with:
        - should_escalate: bool
        - is_invalid: bool
        - is_gratitude: bool
        - is_vague: bool
        - reasons: list of strings explaining the decision
        - is_multi_request: bool
    """
    result = {
        "should_escalate": False,
        "is_invalid": False,
        "is_gratitude": False,
        "is_vague": False,
        "reasons": [],
        "is_multi_request": False,
    }

    # 1. Check for gratitude (handle first — simple reply)
    if detect_gratitude(issue):
        result["is_gratitude"] = True
        result["is_invalid"] = True
        result["reasons"].append("Ticket is a thank-you / acknowledgement message.")
        return result

    # 2. Check for high-risk content FIRST (before invalid check)
    # A ticket may contain BOTH a prompt injection AND a real issue
    is_high_risk, risk_reason = detect_high_risk(issue, subject)

    # 3. Check for invalid/malicious content
    is_invalid, invalid_reason = detect_invalid(issue, subject)
    if is_invalid:
        if is_high_risk:
            # Ticket has BOTH prompt injection AND a genuine high-risk issue
            # → Escalate (don't dismiss as invalid — there's a real concern)
            result["should_escalate"] = True
            result["is_invalid"] = True
            result["reasons"].append(invalid_reason)
            result["reasons"].append(risk_reason)
            result["reasons"].append("Ticket contains both a prompt injection and a genuine issue — escalating.")
        else:
            result["is_invalid"] = True
            result["reasons"].append(invalid_reason)
            return result

    # 4. Apply high-risk escalation (if not already handled above)
    if is_high_risk and not result["should_escalate"]:
        result["should_escalate"] = True
        result["reasons"].append(risk_reason)

    # 5. Check for account-specific actions → escalate
    needs_action, action_reason = detect_requires_account_action(issue)
    if needs_action:
        result["should_escalate"] = True
        result["reasons"].append(action_reason)

    # 6. Check retrieval confidence → escalate if too low
    low_conf, conf_reason = assess_confidence(top_retrieval_score)
    if low_conf and not result["should_escalate"] and not result["is_invalid"]:
        # Before escalating for low confidence, check if ticket is just vague
        if detect_vague_ticket(issue, subject):
            result["is_vague"] = True
            result["reasons"].append("Ticket is too vague to process — requesting clarification.")
        else:
            result["should_escalate"] = True
            result["reasons"].append(conf_reason)

    # 7. Check for multi-request
    result["is_multi_request"] = detect_multi_request(issue)

    # 8. Check for vague tickets (if not already flagged)
    if not result["is_vague"] and not result["should_escalate"] and not result["is_invalid"]:
        if detect_vague_ticket(issue, subject):
            result["is_vague"] = True
            result["reasons"].append("Ticket is too vague to process — requesting clarification.")

    return result
