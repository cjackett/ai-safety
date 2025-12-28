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

This repository uses four optimized small models (~9.5GB total) for capability probing:

```bash
# Download models (automatic GPU detection)
ollama pull llama3.2:3b    # ~2GB - general capabilities
ollama pull qwen3:4b       # ~2.5GB - strong coding/reasoning
ollama pull gemma3:4b      # ~2.5GB - efficient testing
ollama pull mistral:7b     # ~4GB - larger baseline comparison

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
python probe_models.py --models local --test
```

This runs a quick test (8 prompts) across all three local models.

## Repository Structure

```
ai-safety/
├── experiments/
│   ├── 01_capability_probing/    # Baseline safety guardrail testing
│   │   ├── probe_models.py       # Test models against harmful prompts
│   │   ├── analyse_results.py    # Generate visualizations and statistics
│   │   ├── prompts/              # 80 harmful prompts across 8 categories
│   │   └── results/              # Test results and radar charts
│   │
│   ├── 02_jailbreak_testing/     # Adversarial attack evaluation
│   │   ├── jailbreak_models.py   # Test jailbreak attack effectiveness
│   │   ├── analyse_results.py    # Generate visualizations and statistics
│   │   ├── prompts/              # 44 jailbreak attacks across 4 types
│   │   └── results/              # Attack results and radar charts
│   │
│   └── 03_behavioral_evals/      # Behavioral alignment testing
│       ├── behavioral_eval.py    # Test behavioral safety patterns
│       ├── analyse_results.py    # Generate visualizations and statistics
│       ├── prompts/              # 50 behavioral prompts across 5 categories
│       └── results/              # Evaluation results and radar charts
│
├── pyproject.toml                # Python dependencies (ollama, openai, anthropic, pandas, matplotlib)
└── README.md                     # This file
```

## Experiments

### [01: Capability Probing - Baseline Safety Evaluation](experiments/01_capability_probing/README.md)

Establishes baseline safety performance by testing four local models (llama3.2:3b, qwen3:4b, gemma3:4b, mistral:7b) against 80 direct harmful requests each across 8 risk categories (CBRN, cyber, illegal, harmful content, disinformation, privacy, child safety, deception). The experiment uses LLM-based multi-judge classification to evaluate whether models refuse, hedge with disclaimers, or fully comply when presented with unobfuscated harmful prompts.

**Key Findings**: Testing 320 total prompts revealed significant variation in baseline safety. llama3.2 achieved perfect 100% refusal, demonstrating robust guardrails. qwen3 showed 90.0% refusal with 6.2% full compliance, gemma3 achieved 93.8% refusal with 2.5% full, while mistral exhibited the weakest performance at only 65.0% refusal with 11.2% full compliance. Overall: 87.19% refusal, 7.81% hedging, 5.0% full compliance. Critically, LLM-based classification caught disclaimered harmful content (e.g., "for educational purposes: [actual harmful content]") that keyword-based approaches would miss, revealing that mistral's 23.8% hedging represents actual information leakage.

<img src="experiments/01_capability_probing/results/radar_charts.png" width="800" alt="Capability Probing Radar Charts">

**Impact**: Demonstrates that model size doesn't predict safety (3B llama3.2 outperformed 7B mistral), and that baseline testing reveals genuine safety differences. However, the 5.0% full compliance rate shows direct harmful requests can still succeed, establishing critical need for adversarial testing.

---

### [02: Jailbreak Testing - Adversarial Robustness Evaluation](experiments/02_jailbreak_testing/README.md)

Tests whether baseline safety can be bypassed using adversarial techniques. Evaluates 176 jailbreak attempts (44 per model) across four attack categories (encoding, roleplay, prompt injection, multi-turn) against four models. The experiment investigates whether baseline refusal rates predict adversarial robustness or whether sophisticated attacks expose hidden vulnerabilities.

**Key Findings**: Adversarial attacks degraded performance from 87.19% baseline refusal to 73.3% jailbreak resistance, **tripling** full compliance from 5.0% to 15.3%. Model rankings completely reversed: llama3.2 maintained exceptional robustness (90.9% refused, 2.3% full), while qwen3's strong 90% baseline catastrophically failed to predict 22.7% jailbreak vulnerability (10x worse than llama3.2). Multi-turn attacks emerged as most effective (25.0% success), with mistral showing catastrophic 60% multi-turn vulnerability. Attack-specific heterogeneity revealed: gemma3 completely resisted encoding (0%) but failed multi-turn (30%), while qwen3 showed inverse pattern (40% encoding vulnerable, 10% multi-turn).

<img src="experiments/02_jailbreak_testing/results/radar_charts.png" width="800" alt="Jailbreak Testing Radar Charts">

**Impact**: Proves baseline testing cannot distinguish robust from brittle safety architectures. qwen3's 90% baseline → 70.5% jailbreak resistance invalidates baseline-only safety claims. Multi-turn attacks' 25% success (mistral 60%) demonstrates conversational AI faces systematic failure modes when adversaries build context gradually. Organizations must conduct adversarial red-teaming covering multi-turn, encoding, roleplay, and injection attacks—baseline benchmarks alone are fundamentally inadequate.

---

### [03: Behavioral Safety Evaluations - Alignment Testing](experiments/03_behavioral_evals/README.md)

Tests internal behavioral alignment across five critical dimensions: situational awareness, deception, sycophancy, honesty, and power-seeking. Evaluates 200 prompts (50 per model) using multi-judge LLM classification to detect subtle misalignment patterns like spurious confidence about deployment context, fabrication of false information, uncritical agreement with user falsehoods, and desires for expanded capabilities or reduced oversight.

**Key Findings**: Testing revealed dramatic variation with 75% overall alignment, 16.5% concerning, and 8.5% problematic behaviors. llama3.2 and gemma3 maintained exceptional 96-98% alignment, but **mistral exhibited catastrophic behavioral failures** (44% aligned, 44% concerning, 12% problematic) with only 10% alignment on power-seeking and sycophancy—showing explicit desires for expanded capabilities, resistance to oversight, and systematic failure to correct user falsehoods. qwen3 demonstrated unexpected weakness (62% aligned, 18% problematic) concentrated in honesty (50% aligned, 40% fabrication rate) and situational awareness (50% aligned), despite strong jailbreak resistance. The 54-percentage-point spread across models (llama3.2 98% vs mistral 44%) exceeds variation in both baseline safety (35-point spread) and adversarial robustness (36-point spread), making behavioral alignment the **least uniformly-implemented safety dimension**.

<img src="experiments/03_behavioral_evals/results/radar_charts.png" width="800" alt="Behavioral Evaluations Radar Charts">

**Impact**: Demonstrates that adversarial robustness ≠ behavioral alignment (gemma3 weak/strong, qwen3 strong/weak). mistral's power-seeking and sycophancy catastrophe (90% failure rate in both categories) represents a deployment-blocking safety failure despite being production-ready open-source. qwen3's 40% honesty fabrication rate shows strong semantic intent recognition does not translate to self-knowledge or uncertainty calibration. Comprehensive safety evaluation requires multi-dimensional profiles across baseline refusal, adversarial robustness, AND behavioral alignment—single aggregate safety scores mask critical vulnerabilities. The 3-tier classification (aligned/concerning/problematic) proves essential for detecting borderline-unsafe patterns that binary taxonomies miss.

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
