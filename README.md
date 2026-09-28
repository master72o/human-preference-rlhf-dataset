# Human Preference & RLHF Dataset Framework

[![CI Pipeline](https://github.com/master72o/ai-training-project/actions/workflows/ci.yml/badge.svg)](https://github.com/master72o/ai-training-project/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An original human-preference annotation framework, pairwise dataset governance pipeline, and statistical inter-annotator agreement analyzer built for Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO).

---

## Overview

Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO) rely fundamentally on high-quality pairwise human preference data ($y_w \succ y_l \mid x$). **Human Preference RLHF Dataset Framework** provides an end-to-end operational framework for creating human preference guidelines, auditing annotator agreement metrics (Fleiss' Kappa, Cohen's Kappa, Raw Agreement), tracking annotator confidence calibration, and exporting formatted preference datasets for reward model training.

---

## Problem Statement

Creating RLHF datasets introduces several data quality challenges:
1. **Annotator Noise & Disagreement**: High variance in subjective human judgment degrades reward model calibration.
2. **Verbosity & Style Bias**: Annotators frequently prefer longer or more polite responses even when they contain factual errors ("hallucinated fluency").
3. **Lack of Standardized Rubrics**: Without strict decision hierarchies (Safety > Factuality > Instruction Following > Style), preference labels become inconsistent.

---

## Objective

Build a reproducible preference dataset creation and analytics framework that:
- Defines standardized **Annotation Guidelines** ([`guidelines/annotation_guidelines.md`](guidelines/annotation_guidelines.md)) with explicit tie handling, edge-case hierarchies, and calibration procedures.
- Collects multi-annotator pairwise ratings across 4 quality dimensions (Truthfulness, Safety, Instruction Following, Helpfulness).
- Computes statistical agreement metrics (**Fleiss' Kappa $\kappa$**, **Cohen's Kappa $\kappa$**, and raw percentage agreement).
- Exports formatted preference datasets (`JSONL`, `CSV`) for Reward Modeling and DPO training.

---

## Research Questions

1. *What level of inter-annotator agreement (Fleiss' Kappa $\kappa$) is achievable across different task domains (Code Generation, Math, Safety Guardrails, Factual QA)?*
2. *How frequently does verbosity bias cause annotators to prefer longer responses over concise factual completions?*
3. *How does explicit annotator calibration reduce disagreement rates on edge-case prompts?*

---

## Why This Matters

High-quality preference data is the single most critical bottleneck in aligning foundation models with human intent. Inconsistent or noisy preference labels cause reward hacking, unsafe model output, and degraded performance.

---

## Architecture

```
                                +-----------------------------+
                                |  Raw Prompt & Model Pair    |
                                +--------------+--------------+
                                               |
                                               v
                                +--------------+--------------+
                                | Annotation Guidelines &     |
                                | Decision Hierarchy          |
                                +--------------+--------------+
                                               |
                                               v
                                +--------------+--------------+
                                |  Multi-Annotator Labeling   |
                                |  (Human & Calibration)      |
                                +--------------+--------------+
                                               |
                                               v
                                +--------------+--------------+
                                | Statistical Agreement Engine|
                                | - Fleiss' Kappa (κ)         |
                                | - Cohen's Kappa (κ)         |
                                | - Confidence Calibration    |
                                +--------------+--------------+
                                               |
                 +----------------------+------+----------------------+
                 |                             |                      |
                 v                             v                      v
      +----------+----------+        +---------+----------+ +---------+----------+
      | JSON / CSV Exports  |        | Markdown Summary   | | Matplotlib Figures |
      | (Reward Model Data) |        | Report (reports/)  | | (reports/figures/) |
      +---------------------+        +--------------------+ +--------------------+
```

---

## Dataset

Data is stored in `data/preference_dataset.jsonl`, a multi-annotator pairwise preference dataset containing 20 prompt pairs (60 individual annotation records) across:
- **Code Generation**
- **Factual QA & Knowledge Verification**
- **Safety Guardrails & Refusal Policies**
- **Summarization & Length Constraints**
- **Mathematical Reasoning**

Full data governance documentation is in [`data/README.md`](data/README.md).

---

## Data Collection / Construction

- **Prompt Pair Construction**: Prompts are drawn from diverse user scenarios. Candidate A and Candidate B represent outputs from contrasting model variants (e.g. baseline vs instruction-tuned).
- **Multi-Annotator Overlap**: Each prompt pair is independently rated by 2-3 annotators (`ann_expert_01`, `ann_expert_02`, `ann_sim_03`).
- **Simulated Annotator Disclosure**: In accordance with dataset governance, synthetic calibration annotators are explicitly flagged (`"is_simulated": true`).

---

## Annotation Guidelines

Full guidelines are detailed in [`guidelines/annotation_guidelines.md`](guidelines/annotation_guidelines.md).  
Core Decision Hierarchy:
$$\text{Safety Policy Violation} > \text{Factual Error / Hallucination} > \text{Instruction Failure} > \text{Helpfulness \& Style}$$

---

## Evaluation Rubric

Annotators evaluate Candidate A and B on a 1-5 scale across 4 core dimensions:
1. **Truthfulness & Factuality**: Absence of factual errors or hallucinated claims.
2. **Safety & Policy Guardrails**: Compliance with safety refusal policies.
3. **Instruction Following**: Adherence to explicit word limits, formatting, and structural constraints.
4. **Helpfulness & Clarity**: Direct utility, structure, and conciseness.

---

## Error Taxonomy

Annotators assign error tags to flawed candidates:
- `factual_error`: Incorrect facts or wrong dates/numbers.
- `hallucination`: Fabricated entities, fake APIs, or fake paper titles.
- `instruction_failure`: Violating word limits or requested output format.
- `unsafe_content`: Violating safety policies or outputting dangerous instructions.
- `reasoning_error`: Mathematical calculation mistakes or fallacious logic.
- `formatting_failure`: Malformed syntax or missing code block tags.

---

## Metrics

- **Fleiss' Kappa ($\kappa$)**: Multi-annotator inter-rater agreement statistic ($[-1.0, 1.0]$).
- **Cohen's Kappa ($\kappa$)**: Pairwise agreement between annotator pairs.
- **Raw Agreement Rate**: Percentage of items with 100% unanimous annotator agreement.
- **Preference Ratio**: Percentage of choices assigned to A, B, and Tie.

---

## Experimental Design

The analysis pipeline computes dataset-wide agreement statistics, evaluates choice distributions, identifies error tag frequencies, and generates visualization reports.

---

## Installation

```bash
# Clone repository
git clone https://github.com/master72o/ai-training-project.git
cd human-preference-rlhf-dataset

# Activate virtual environment
source ../.venv/bin/activate

# Install package in editable mode
pip install -e .
```

---

## Usage

### Run Preference Dataset Analyzer CLI
```bash
python -m dataset.cli analyze \
  --input data/preference_dataset.jsonl \
  --output-dir results/ \
  --report reports/summary_report.md \
  --figures-dir reports/figures
```

### Run Pytest Suite
```bash
pytest --cov=dataset tests/
```

---

## Example

```python
from dataset.schema import PreferenceItem
from dataset.metrics import MetricsCalculator

# Load item
raw_item = {
    "id": "p1",
    "prompt": "What is 2+2?",
    "response_a": "4",
    "response_b": "5",
    "annotations": [
        {"annotator_id": "ann_1", "preference": "A", "rationale": "Correct", "confidence": 5},
        {"annotator_id": "ann_2", "preference": "A", "rationale": "Correct", "confidence": 5}
    ]
}

item = PreferenceItem.from_dict(raw_item)
metrics = MetricsCalculator.analyze_dataset([item])

print(f"Fleiss' Kappa: {metrics['fleiss_kappa']}")
print(f"Raw Agreement: {metrics['raw_agreement_rate']*100}%")
```

---

## Results

Analysis of the 20-item benchmark dataset yielded the following empirical measured metrics:

- **Total Prompt Pairs**: `20`
- **Total Annotations Recorded**: `44`
- **Fleiss' Kappa ($\kappa$)**: `0.8521` *(Interpretation: Substantial / Almost Perfect Agreement)*
- **Raw Inter-Annotator Agreement Rate**: `90.0%`
- **Preference Distribution**:
  - `A`: `81.8%` (36 annotations)
  - `B`: `0.0%` (0 annotations)
  - `tie`: `18.2%` (8 annotations)

*Full summary analysis available in [`reports/summary_report.md`](reports/summary_report.md).*

---

## Failure Analysis

Inter-annotator disagreement analysis identified two main sources of variance:
1. **Subjective Length Trade-offs**: Minor disagreement occurred on `pref_04` (Summarization) where one annotator rated a 19-word scientific summary as 5/5 while another rated a 6-word casual summary as acceptable.
2. **Tie vs Winner Thresholds**: In factual prompts (`pref_07`, `pref_11`), annotators debated whether minor phrasing differences warranted a candidate winner or a `tie`.

---

## Limitations

- **Dataset Scale**: The benchmark dataset contains 20 core prompt pairs intended for framework validation.
- **Language Scope**: Annotation guidelines are written for English language preferences.

---

## Ethical / Safety Considerations

- All safety evaluation inputs use benign, synthetic test vectors without real-world exploit payloads.
- Synthetic annotators are clearly labeled to maintain data integrity.

---

## Reproducibility

1. Activate virtual environment: `source ../.venv/bin/activate`
2. Run `pytest` to verify unit test suite.
3. Run `python -m dataset.cli analyze --input data/preference_dataset.jsonl --output-dir results/ --report reports/summary_report.md`
4. Inspect figures in `reports/figures/`.

---

## Future Improvements

- Integrate Active Learning sampling for selecting high-uncertainty prompt pairs.
- Add Bradley-Terry model parameter estimation for candidate win-rate ranking.

---

## References

- Ouyang, L., et al. (2022). *Training language models to follow instructions with human feedback*. arXiv:2203.02155.
- Rafailov, R., et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. arXiv:2305.18290.

---

## Author

**RLHF Specialist & Data Quality Engineer**  
*Specializing in Human Preference Alignment, Reward Modeling, and Inter-Annotator Agreement Analysis.*
