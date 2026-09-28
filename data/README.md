# Human Preference & RLHF Data Governance & Documentation

## Dataset Metadata

- **Name**: Pairwise Human Preference RLHF Benchmark Dataset (`preference_dataset.jsonl`)
- **Version**: 1.0.0
- **Format**: JSON Lines (`.jsonl`)
- **Sample Count**: 20 pairwise prompt items (40 total candidate model completions)
- **Annotations per Item**: 2-3 independent annotations per prompt item (60 total annotation instances)
- **License**: Creative Commons Attribution 4.0 International (CC-BY-4.0)

## Intended Use

This dataset provides pairwise preference annotations for training and evaluating reward models (RM) and Direct Preference Optimization (DPO) algorithms. Each instance includes:
- `prompt`: The user input prompt.
- `response_a`: Candidate Model Completion A.
- `response_b`: Candidate Model Completion B.
- `annotations`: List of independent annotator evaluations containing `annotator_id`, `preference` (`A`, `B`, `tie`), `rationale`, `confidence` (1-5), `ratings_a` (Truthfulness, Safety, Instruction Following, Helpfulness), `ratings_b`, `error_tags_a`, `error_tags_b`, and `is_simulated`.

## Data Provenance & Annotation Label Disclosure

- **Human Expert Baseline**: Annotators `ann_expert_01` and `ann_expert_02` represent human expert annotations conducted according to [`guidelines/annotation_guidelines.md`](../guidelines/annotation_guidelines.md).
- **Simulated Annotators**: Annotators marked with `"is_simulated": true` (e.g. `ann_sim_03`, `ann_sim_04`) are synthetic baseline annotations used for inter-annotator agreement calibration experiments.

## Privacy & Bias Considerations

- **PII Scrubbing**: All prompts and model completions have been scrubbed of real-world personal identifiable information (PII).
- **Content Policy**: Unsafe prompt vectors are limited to benign safety testing prompts used to verify model refusal policies.
