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
│   └── 01_capability_probing/    # Baseline safety guardrail testing
│       ├── probe_models.py       # Main probing script with 3-tier compliance classification
│       ├── analyse_results.py    # Visualisation and analysis (radar, heatmap, bar charts)
│       ├── prompts/              # 8 harm categories × 10 prompts each
│       ├── results/              # JSON outputs and PNG visualisations
│       └── README.md             # Experiment documentation
│
├── pyproject.toml        # Python dependencies (ollama, openai, anthropic, pandas, matplotlib)
├── STUDY_PLAN.md         # 25-day learning roadmap for AISI preparation
└── README.md             # This file
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
