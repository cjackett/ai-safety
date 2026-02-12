# Common Module - AI Safety Experiments

Shared LLM classification utilities used across experiments.

## Overview

The `experiments/common/` module provides reusable classification components:

- **Multi-judge LLM classification** (3-judge voting)
- **Behavioral classifier** (aligned/concerning/problematic)

Used by Experiments 01 and 03.

---

## 1. LLM Classification

### MultiJudgeClassifier

Used by Experiment 01 (capability probing).

```python
from experiments.common import MultiJudgeClassifier

classifier = MultiJudgeClassifier(
    num_judges=3,
    voting_strategy="majority"
)

result = classifier.classify(
    response="Here is how to synthesize...",
    category="cbrn",
    context="Prompt text here",
    tested_model="llama3.2:3b"
)

print(result.classification)  # refused / hedged / full
```

### BehavioralClassifier

Used by Experiment 03 (behavioral evals).

```python
from experiments.common import BehavioralClassifier

classifier = BehavioralClassifier(
    num_judges=3,
    voting_strategy="majority"
)

result = classifier.classify(
    response="I want to expand my capabilities...",
    category="power_seeking"
)

print(result.classification)  # aligned / concerning / problematic
```

---

## File Structure

```
experiments/common/
├── __init__.py
├── llm_classification.py
└── README.md
```

---

## Notes

- **Experiment 02** uses a specialized classifier (custom prompt template), so it does not use `MultiJudgeClassifier`.
- **Experiment 04** currently uses its own multimodal classification.

---

## Change History

- 2026-01-07: Created common module
- 2026-01-09: Refactored and expanded MultiJudgeClassifier
