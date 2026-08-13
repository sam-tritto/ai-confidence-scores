# Enterprise AI Confidence Calibration Suite (`ai-confidence-scores`)

> Production-Grade Evaluation, Calibration, and Benchmarking of 10 Distinct LLM Confidence Score Frameworks for Multimodal PDF Resume Parsing & Classification using Google's `google-genai` SDK.

---

## Executive Summary & Case Study

When deploying Large Language Models (LLMs) like **Gemini 2.5 Flash** for automated resume screening and candidate domain classification, raw model outputs frequently suffer from **probability saturation** and **overconfidence**. Uncalibrated confidence scores can lead to silent misclassifications, invalid automated hiring routes, or excessive manual auditing work.

This repository provides an enterprise-grade framework that evaluates, calibrates, and benchmarks **10 distinct confidence scoring techniques** on PDF resumes. It introduces a dual PDF ingestion pipeline (layout-preserved PyMuPDF text parsing + native Gemini multimodal PDF byte ingestion) and routes extractions dynamically between **`AUTOMATE`** and **`FLAG_FOR_HUMAN_REVIEW`**.

---

## System Architecture

```
                                ┌───────────────────────────────┐
                                │   PDF Resume Input Dataset    │
                                └──────────────┬────────────────┘
                                               │
                                               ▼
                                ┌───────────────────────────────┐
                                │  Dual Ingestion Pipeline      │
                                │  • PyMuPDF Layout Extractor   │
                                │  • Gemini Multimodal Bytes    │
                                └──────────────┬────────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                             10 Calibration & Scoring Methods                                │
├───────────────────────────────┬───────────────────────────────┬─────────────────────────────┤
│ 1. Native Token LogProb       │ 2. LogProb Delta              │ 3. Temperature Scaling      │
│ 4. Platt Scaling Logistic     │ 5. Self-Consistency Agreement │ 6. Structured Self-Assess.  │
│ 7. Continuous Prompting       │ 8. Grounding & Alignment      │ 9. LLM-as-a-Judge           │
│ 10. Self-Assess + Platt       │                               │                             │
└───────────────────────────────┴──────────────┬────────────────┴─────────────────────────────┘
                                               │
                                               ▼
                                ┌───────────────────────────────┐
                                │  Evaluation & Audit Router    │
                                │  • ECE / MCE / Brier Score    │
                                │  • AUTOMATE vs HUMAN_REVIEW   │
                                └───────────────────────────────┘
```

---

## Mathematical Breakdown of 10 Confidence Frameworks

### 1. Native Token LogProb Method (`NativeLogProbMethod`)
Configures `response_logprobs=True` and `logprobs=5` in `GenerateContentConfig`. Computes joint sequence confidence as the geometric mean token probability:
$$\text{Confidence} = \exp\left( \frac{1}{N} \sum_{i=1}^N \text{logprob}_i \right)$$

### 2. LogProb Delta Method (`LogProbDeltaMethod`)
Measures top-token dominance margin at critical classification tokens to eliminate probability saturation:
$$\Delta = \text{logprob}_{\text{top\_1}} - \text{logprob}_{\text{top\_2}}$$
$$\text{Confidence} = \sigma(\Delta) = \frac{1}{1 + e^{-\Delta}}$$

### 3. Post-Hoc Temperature Scaling Method (`TemperatureScalingMethod`)
Rescales logits $z_i$ via a temperature parameter $T > 1$ fitted using Negative Log-Likelihood (NLL) optimization:
$$P_{\text{calibrated}}(i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

### 4. Platt Scaling Logistic Method (`PlattScalingMethod`)
Trains a 1D logistic regression model mapping logprob delta or raw confidence to empirical correctness:
$$P_{\text{calibrated}} = \sigma(a \cdot \Delta + b) = \frac{1}{1 + e^{-(a \cdot \Delta + b)}}$$

### 5. Self-Consistency & Sampling Agreement Method (`SelfConsistencyMethod`)
Generates $N=5$ parallel completions at temperature $T=0.7$, clusters outputs, and computes the agreement ratio of the majority cluster:
$$\text{Confidence} = \frac{\text{Majority Count}}{N}$$

### 6. Structured Self-Assessment Method (`VerbalizedConfidenceMethod`)
Requests step-by-step rationale and an explicit verbalized confidence score between $0.0$ and $1.0$ alongside structured extraction data.

### 7. Continuous Numerical Prompting Method (`ContinuousPromptingMethod`)
Prompts the LLM for a continuous rating $R \in [0, 100]$ alongside chain-of-thought rationale, normalizing to:
$$\text{Confidence} = \frac{R}{100.0}$$

### 8. Grounding & Context Alignment Method (`GroundingAlignmentMethod`)
Extracts atomic factual claims from the output and performs Natural Language Inference (NLI) verification against source PDF resume text:
$$\text{Grounding Score} = \frac{\text{Supported Claims}}{\text{Total Claims}}$$

### 9. Two-Tier Evaluator Method (`LLMAsAJudgeMethod`)
Submits candidate extractions to a secondary judge instance (`gemini-2.5-flash`) with an evaluation rubric scoring precision, hallucination risk, and completeness (1–5 scale), normalized to $[0.0, 1.0]$.

### 10. Structured Self-Assessment + Platt Scaling Method (`StructuredSelfAssessmentPlattMethod`)
Combines structured LLM self-assessment (categorical levels or continuous $0.0-1.0$) with post-hoc **Platt Scaling** (logistic calibration). Converts raw self-assessment probability $p_{\text{raw}}$ to logit $z = \text{logit}(p_{\text{raw}})$, then applies logistic scaling fitted on validation data via NLL loss:
$$P_{\text{calibrated}} = \sigma(a \cdot z + b) = \frac{1}{1 + e^{-(a \cdot z + b)}}$$
*Note: Fully logprob-independent, designed specifically for Gemini > 3.0 models where token logprobs are unavailable or disabled.*

---

## Installation & Setup

This repository uses [`uv`](https://github.com/astral-sh/uv) for fast, reproducible dependency management.

```bash
# 1. Clone repository
git clone https://github.com/sam-tritto/ai-confidence-scores.git
cd ai-confidence-scores

# 2. Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# 3. Configure environment variables (Copy .env.example)
cp .env.example .env
# Edit .env and insert your GEMINI_API_KEY
```

---

## Quickstart Code Example

```python
from src.ingestion import ResumeIngestor
from src.calibration import LogProbDeltaMethod, GroundingAlignmentMethod, StructuredSelfAssessmentPlattMethod

# 1. Ingest PDF Resume
ingestor = ResumeIngestor(data_directory="./data")
resume_text, pdf_bytes = ingestor.process_pdf("./data/10399912.pdf")

# 2. Evaluate with Structured Self-Assessment + Platt Scaling Method
method = StructuredSelfAssessmentPlattMethod()
result = method.evaluate(resume_text, pdf_bytes=pdf_bytes)

print(f"Method: {result.method_name}")
print(f"Candidate Name: {result.extraction.candidate_name}")
print(f"Domain Role: {result.extraction.domain_role.value}")
print(f"Calibrated Confidence: {result.calibrated_confidence:.3f}")
print(f"Audit Routing Decision: {result.audit_decision.value}")
```

---

## Running Unit Tests

Execute the full `pytest` suite with unit tests for schemas, ingestion, calibration methods, and metrics:

```bash
uv run pytest -v
```

---

## Jupyter Notebook Tutorial

Launch or run the complete tutorial notebook demonstrating all 10 methods and generating visual charts:

```bash
uv run jupyter notebook notebooks/calibration_benchmark_tutorial.ipynb
```

---

## License
MIT License. Developed for enterprise AI engineering benchmarking.
