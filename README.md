# AI Safety Engineering Resources

## Overview

This repository contains resources, experiments, and tools related to AI safety evaluation engineering, with emphasis on frontier model evaluation, red-teaming frameworks, behavioral testing, safety tooling, and mechanistic interpretability.

## Repository Structure

```
ai-safety/
├── experiments/          # Experimental implementations
│   ├── 01_capability_probing/
│   ├── 02_jailbreak_testing/
│   ├── 03_behavioral_evals/
│   ├── 04_multimodal_safety/
│   ├── 05_induction_heads/
│   ├── 06_guardrail_testing/
│   └── 07_scaling_evaluation/
│
├── evaluations/          # Production-quality evaluation frameworks
│   ├── safetybench/
│   └── priority_evals/
│
├── interpretability/     # Mechanistic interpretability implementations
│   ├── circuit_discovery/
│   └── sae_experiments/
│
├── papers/              # Literature notes and summaries
│
├── portfolio/           # Reference implementations
│   ├── 1_evaluation_framework/
│   ├── 2_multimodal_safety_eval/
│   ├── 3_red_teaming_suite/
│   ├── 4_safety_pipeline/
│   ├── 5_interpretability_analysis/
│   └── 6_safety_assessment_report/
│
└── resources/           # Curated resources and documentation
```

## Technical Stack

**Core Libraries**:
- `transformers`, `torch`, `pytorch-lightning` - Model training and inference
- `openai`, `anthropic` - Frontier model APIs
- `inspect-ai` - UK AISI evaluation framework
- `garak` - LLM vulnerability scanner
- `transformer-lens` - Mechanistic interpretability
- `pytest`, `black`, `ruff`, `mypy` - Development tools

**Infrastructure**:
- Python 3.12+
- Docker (reproducible environments)
- PostgreSQL (evaluation results storage)

## Key Areas

### Evaluation Engineering
- Automated evaluation frameworks and pipelines
- Behavioral testing methodologies
- Large-scale evaluation infrastructure
- Database backends for result management

### Red-Teaming & Adversarial Testing
- Jailbreak detection and testing suites
- Prompt injection techniques
- Multi-turn attack patterns
- Automated vulnerability scanning

### Mechanistic Interpretability
- Circuit discovery implementations
- Sparse autoencoder experiments
- Activation patching and analysis
- Feature visualization

### Safety Infrastructure
- Guardrails and filtering systems
- Access control implementations
- Inference-time controls
- Continuous monitoring approaches

## Documentation Standards

Each experiment and implementation includes:
- **Comprehensive README**: Motivation, methodology, results, and analysis
- **Clean Code**: Type hints, error handling, and clear documentation
- **Reproducibility**: Version-controlled dependencies and structured data
- **Critical Analysis**: Discussion of results, limitations, and future work

## Resources

- [Australian AI Action Plan](https://www.industry.gov.au/publications/australias-artificial-intelligence-action-plan)
- [UK AI Safety Institute](https://www.aisi.gov.uk/)
- [Anthropic Research](https://www.anthropic.com/research)
- [Inspect AI Documentation](https://ukgovernmentbeis.github.io/inspect_ai/)
- [TransformerLens](https://github.com/neelnanda-io/TransformerLens)
- [Garak LLM Scanner](https://github.com/leondz/garak)
