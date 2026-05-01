# Multi-Domain Support Triage Agent

## Overview

This project implements a **terminal-based support triage agent** that processes customer support tickets across multiple ecosystems:

* HackerRank
* Claude
* Visa

The agent analyzes each ticket, classifies the issue, detects risk, retrieves relevant documentation, and decides whether to **reply or escalate**.

---

## Features

* ✅ Request type classification
  (`product_issue`, `bug`, `feature_request`, `invalid`)

* ✅ Product area detection
  (billing, fraud, account_access, API, privacy, etc.)

* ✅ Risk assessment

  * Fraud detection
  * Billing disputes
  * Account access issues

* ✅ Intelligent decision making

  * Reply for safe/FAQ cases
  * Escalate for high-risk cases

* ✅ TF-IDF based retrieval
  Retrieves relevant support documentation from corpus

* ✅ Grounded response generation
  Ensures responses are based only on provided support data

---

## Architecture

The system follows a modular pipeline:

1. **Input Processing**
   Read tickets from CSV

2. **Classification**
   Determine request type and product area

3. **Risk Detection**
   Identify high-risk scenarios

4. **Retrieval**
   Fetch relevant documents using TF-IDF

5. **Decision Engine**
   Decide whether to reply or escalate

6. **Response Generation**
   Generate safe and grounded responses

7. **Output Generation**
   Save results to output CSV

---

## Project Structure

```plaintext
code/
  main.py
  classifier.py
  corpus_loader.py
  risk_detector.py
  response_generator.py
  config.py
  requirements.txt
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the agent

```bash
python code/main.py
```

---

## Input

The agent reads support tickets from:

```plaintext
support_tickets/support_tickets.csv
```

---

## Output

The processed results are saved to:

```plaintext
support_tickets/output.csv
```

Each row contains:

* `status` → replied / escalated
* `product_area`
* `response`
* `justification`
* `request_type`

---

## Decision Logic

### Escalated when:

* Fraud or unauthorized access detected
* Billing disputes or payment issues
* Account access problems
* Security vulnerabilities
* Low confidence in retrieval

### Replied when:

* FAQ or general queries
* Known product issues
* Valid feature requests
* Supported documentation available

---

## Technologies Used

* Python
* pandas
* scikit-learn (TF-IDF)
* Rule-based classification

---

## Notes

* No external APIs are used
* All responses are grounded in the provided support corpus
* High-risk issues are safely escalated
* Designed for deterministic and reliable behavior

---

## Author

Abhienaya Sri
