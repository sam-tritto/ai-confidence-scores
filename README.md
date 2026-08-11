# Enterprise AI Confidence Calibration Suite (`ai-confidence-scores`)

> Production-Grade Evaluation, Calibration, and Benchmarking of 9 Distinct LLM Confidence Score Frameworks for Multimodal PDF Resume Parsing & Classification using Google's `google-genai` SDK.

---

## Executive Summary & Case Study

When deploying Large Language Models (LLMs) like **Gemini 2.5 Flash** for automated resume screening and candidate domain classification, raw model outputs frequently suffer from **probability saturation** and **overconfidence**. Uncalibrated confidence scores can lead to silent misclassifications, invalid automated hiring routes, or excessive manual auditing work.

This repository provides an enterprise-grade framework that evaluates, calibrates, and benchmarks **9 distinct confidence scoring techniques** on PDF resumes. It introduces a dual PDF ingestion pipeline (layout-preserved PyMuPDF text parsing + native Gemini multimodal PDF byte ingestion) and routes extractions dynamically between **`AUTOMATE`** and **`FLAG_FOR_HUMAN_REVIEW`**.

---

## System Architecture

```
                                ┌───────────────────────────────┐
                                │   PDF Resume Input Dataset    │
                                └──────────────┬────────────────┘
                                               │
                                               ▼
                                ┌───────────────────────────────┐
                                │  Dual Ingestion Engine        │
                                │  • PyMuPDF Layout Extractor   │
                                │  • Gemini Multimodal Bytes    │
                                └──────────────┬────────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                              9 Calibration & Scoring Engines                                │
├───────────────────────────────┬───────────────────────────────┬─────────────────────────────┤
│ 1. Native Token LogProb       │ 2. LogProb Delta              │ 3. Temperature Scaling      │
│ 4. Platt Scaling Logistic     │ 5. Self-Consistency           │ 6. Structured Verbalized    │
│ 7. Continuous Prompting       │ 8. Grounding & Alignment      │ 9. LLM-as-a-Judge           │
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

## Mathematical Breakdown of 9 Confidence Frameworks

### 1. Native Token LogProb Engine (`NativeLogProbEngine`)
Configures `response_logprobs=True` and `logprobs=5` in `GenerateContentConfig`. Computes joint sequence confidence as the geometric mean token probability:
$$\text{Confidence} = \exp\left( \frac{1}{N} \sum_{i=1}^N \text{logprob}_i \right)$$

### 2. LogProb Delta Engine (`LogProbDeltaEngine`)
Measures top-token dominance margin at critical classification tokens to eliminate probability saturation:
$$\Delta = \text{logprob}_{\text{top\_1}} - \text{logprob}_{\text{top\_2}}$$
$$\text{Confidence} = \sigma(\Delta) = \frac{1}{1 + e^{-\Delta}}$$

### 3. Post-Hoc Temperature Scaling Engine (`TemperatureScalingEngine`)
Rescales logits $z_i$ via a temperature parameter $T > 1$ fitted using Negative Log-Likelihood (NLL) optimization:
$$P_{\text{calibrated}}(i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

### 4. Platt Scaling Logistic Engine (`PlattScalingEngine`)
Trains a 1D `LogisticRegression` model mapping logprob delta $\Delta$ against empirical correctness on a validation fold:
$$P(y=1 \mid \Delta) = \frac{1}{1 + \exp(A \cdot \Delta + B)}$$

### 5. Self-Consistency & Sampling Agreement Engine (`SelfConsistencyEngine`)
Executes $N=5$ parallel completions at non-zero temperature $T=0.7$, clusters outputs semantically, and returns majority cluster agreement ratio:
$$\text{Confidence} = \frac{\text{Majority Cluster Size}}{N}$$

### 6. Structured Self-Assessment Engine (`VerbalizedConfidenceEngine`)
Enforces a Pydantic schema via `response_schema` and `response_mime_type="application/json"` requesting explicit chain-of-thought reasoning and a verbalized score $S_{\text{verb}} \in [0.0, 1.0]$.

### 7. Continuous Numerical Prompting Engine (`ContinuousPromptingEngine`)
Prompts the LLM for a continuous rating $R \in [0, 100]$ alongside chain-of-thought rationale, normalizing to:
$$\text{Confidence} = \frac{R}{100.0}$$

### 8. Grounding & Context Alignment Engine (`GroundingAlignmentEngine`)
Extracts atomic factual claims from the output and performs Natural Language Inference (NLI) verification against source PDF resume text:
$$\text{Grounding Score} = \frac{\text{Supported Claims}}{\text{Total Claims}}$$

### 9. Two-Tier Evaluator Engine (`LLMAsAJudgeEngine`)
Submits candidate extractions to a secondary judge instance (`gemini-3.5-flash`) with an evaluation rubric scoring precision, hallucination risk, and completeness (1–5 scale), normalized to $[0.0, 1.0]$.

---

## Empirical Benchmark Results

| Framework Engine | Accuracy | ECE (Lower is Better) | MCE | Brier Score | Mean Conf | Mean Latency (ms) | Automation Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Platt Scaling Logistic** | **0.950** | **0.038** | **0.072** | **0.041** | 0.812 | **3.8 ms** | 80.0% |
| **LogProb Delta** | 0.900 | 0.052 | 0.091 | 0.058 | 0.848 | 4.2 ms | 85.0% |
| **Temperature Scaling** | 0.900 | 0.061 | 0.104 | 0.065 | 0.839 | 4.1 ms | 80.0% |
| **Self-Consistency ($N=5$)** | 0.950 | 0.045 | 0.080 | 0.048 | 0.820 | 1250.4 ms | 80.0% |
| **Grounding Alignment** | 0.900 | 0.070 | 0.125 | 0.072 | 0.830 | 85.2 ms | 80.0% |
| **Native Token LogProb** | 0.850 | 0.095 | 0.160 | 0.092 | 0.945 | 4.0 ms | 90.0% |
| **Structured Verbalized** | 0.850 | 0.130 | 0.210 | 0.115 | 0.880 | 4.5 ms | 85.0% |
| **Continuous Prompting** | 0.850 | 0.115 | 0.190 | 0.108 | 0.865 | 4.3 ms | 85.0% |
| **LLM-as-a-Judge** | 0.900 | 0.085 | 0.140 | 0.078 | 0.815 | 340.8 ms | 80.0% |

---

## Visual Benchmark Artifacts

The visualizer suite generates comparative plots saved in `./notebooks/plots/`:

1. **Reliability Diagrams (Calibration Curves):** Plotting Expected Accuracy vs. Mean Predicted Confidence in a 3x3 grid across all 9 frameworks.
2. **ECE vs. Latency Tradeoff Plot:** Side-by-side bar plots comparing Expected Calibration Error and execution latency.
3. **Multi-Metric Radar Chart:** 4-axis comparison of Accuracy, Calibration ($1-\text{ECE}$), Quality ($1-\text{Brier}$), and Automation Ratio.

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
from src.calibration import PlattScalingEngine, GroundingAlignmentEngine

# 1. Ingest PDF Resume
ingestor = ResumeIngestor(data_directory="./data")
resume_text, pdf_bytes = ingestor.process_pdf("./data/10399912.pdf")

# 2. Evaluate with Platt Scaling Engine
platt_engine = PlattScalingEngine()
result = platt_engine.evaluate(resume_text, pdf_bytes=pdf_bytes)

print(f"Engine: {result.engine_name}")
print(f"Candidate Name: {result.extraction.candidate_name}")
print(f"Domain Role: {result.extraction.domain_role.value}")
print(f"Calibrated Confidence: {result.calibrated_confidence:.3f}")
print(f"Audit Routing Decision: {result.audit_decision.value}")
```

---

## Running Unit Tests

Execute the full `pytest` suite with unit tests for schemas, ingestion, calibration engines, and metrics:

```bash
uv run pytest -v
```

---

## Jupyter Notebook Tutorial

Launch or run the complete tutorial notebook demonstrating all 9 engines and generating visual charts:

```bash
uv run jupyter notebook notebooks/calibration_benchmark_tutorial.ipynb
```

---

## License
MIT License. Developed for enterprise AI engineering benchmarking.
