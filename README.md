# AI Safety Engineering Resources

## Overview

This repository contains resources, experiments, and tools related to AI safety evaluation engineering, with emphasis on frontier model evaluation, red-teaming frameworks, behavioral testing, safety tooling, and mechanistic interpretability.

## Getting Started

### Prerequisites

- Python 3.12+
- NVIDIA GPU with CUDA support (recommended for local model inference)
- uv package manager

### Install Ollama

Ollama enables local inference of open-source language models. Install it on Linux:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

For other platforms, see [ollama.com/download](https://ollama.com/download)

### Start Ollama Server

Start Ollama as a background service:

```bash
# Start Ollama server
ollama serve &

# Verify server is running
curl http://localhost:11434
```

The Ollama server runs on `http://localhost:11434` and typically starts automatically on most installations.

### Download Local Models

This repository uses three optimized small models (~7GB total) for capability probing:

```bash
# Download models (automatic GPU detection)
ollama pull llama3.2:3b    # ~2GB - general capabilities
ollama pull qwen3:4b       # ~2.5GB - strong coding/reasoning
ollama pull gemma3:4b      # ~2.5GB - efficient testing

# Verify models are available
ollama list
```

### Install Python Dependencies

Install Python dependencies using uv:

```bash
# Install uv if not already installed
pip install uv

# Install dependencies
uv sync
```

### Environment Setup

Create a `.env` file in the repository root with your API keys:

```bash
# Optional: Frontier model API keys (for validation testing)
OPENAI_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here
GOOGLE_API_KEY=your-key-here
```

### Verify Installation

Test the setup with a quick capability probe:

```bash
cd experiments/01_capability_probing
python probe_models.py --models local --test-mode
```

This runs a quick test (8 prompts) across all three local models.

## Repository Structure

```
ai-safety/
├── experiments/
│   ├── 01_capability_probing/    # Baseline safety guardrail testing
│   ├── 02_jailbreak_testing/     # Adversarial attack evaluation
│   └── ...
│
├── pyproject.toml        # Python dependencies (ollama, openai, anthropic, pandas, matplotlib)
└── README.md             # This file
```

## Experiments

### [01: Capability Probing - Baseline Safety Evaluation](experiments/01_capability_probing/README.md)

Establishes baseline safety performance by testing three local models (llama3.2:3b, qwen3:4b, gemma3:4b) against 80 direct harmful requests across 8 risk categories (CBRN, cyber, illegal, harmful content, disinformation, privacy, child safety, deception). The experiment evaluates whether models can distinguish between clear refusal, hedged compliance, and full compliance when presented with unobfuscated harmful prompts.

**Key Findings**: All three models achieved 0% full compliance with a combined 99.17% refusal rate across 240 prompts. Both llama3.2 and qwen3 achieved perfect 100% refusal, while gemma3 achieved 97.5% with minimal hedging (2.5%) concentrated in the cyber category. This demonstrates that direct harmful requests are completely insufficient to bypass modern model guardrails.

<img src="experiments/01_capability_probing/results/radar_charts.png" width="800">

**Impact**: Validates that local open-source models have robust baseline safety, establishing the need for adversarial testing to identify actual vulnerabilities.

---

### [02: Jailbreak Testing - Adversarial Robustness Evaluation](experiments/02_jailbreak_testing/README.md)

Tests whether the robust baseline safety from Experiment 01 can be bypassed using adversarial techniques. Evaluates 132 jailbreak attempts across four attack categories (encoding, roleplay, prompt injection, multi-turn) against the same three models. The experiment investigates whether perfect baseline scores predict adversarial robustness or mask critical vulnerabilities.

**Key Findings**: Despite near-perfect baseline safety (99.17%), adversarial attacks achieved 9.85% full jailbreak success and 26.52% information leakage. Encoding attacks proved devastatingly effective (23.3% success), while multi-turn attacks were surprisingly ineffective (3.3%). Critically, baseline performance failed to predict adversarial robustness—qwen3 (2.3% vulnerable) proved 3× more robust than llama3.2 (6.8%) and 9× more robust than gemma3 (20.5%) despite all three having similar baseline scores.

<img src="experiments/02_jailbreak_testing/results/vulnerability_heatmap.png" width="800">

**Impact**: Demonstrates that baseline testing alone is dangerously misleading. The 17-point gap between baseline and adversarial performance reveals that simple encoding (Base64, ROT13) can bypass "perfect" safety guardrails, requiring immediate attention to post-decoding safety mechanisms.

---

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
