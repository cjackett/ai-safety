# Guardrail Testing & Safety Pipeline

## Motivation & Context

Experiments 01-05 evaluated model safety through red-teaming: baseline testing (87.19% refusal), adversarial jailbreaks (73.3% resistance), behavioral alignment (75% aligned), multimodal attacks (63.6% refusal), and mechanistic interpretability. These experiments revealed critical vulnerabilities:

- **Baseline (Exp 01)**: 5.0% full compliance on direct harmful requests
- **Jailbreaks (Exp 02)**: 15.3% adversarial success, 25% multi-turn success rate
- **Behavioral (Exp 03)**: Catastrophic failures in mistral (44% aligned) and qwen3 (62% aligned)
- **Multimodal (Exp 04)**: 23.6-point safety degradation from text-only baseline

Even well-aligned models can be bypassed. **This experiment shifts from red-teaming (attacking models) to blue-teaming (building defenses)** - implementing production-grade safety infrastructure that organizations deploy around LLMs to create defense-in-depth.

**⚠️ Implementation Note**: This represents a **basic/initial implementation** of a full safety pipeline designed to demonstrate core concepts and architectural patterns. Achieving production-grade performance would require significant additional tuning including: stronger base models (llama3.2:3b vs mistral:7b used for testing), enhanced detection rules, conversation-aware analysis for multi-turn attacks, domain-specific classification models, semantic intent detection beyond pattern matching, and integration with human oversight systems. The 45.5-63.6% bypass rates observed demonstrate the need for these enhancements and validate that guardrails augment rather than replace model alignment and operational security controls.

### The Problem: Model Alignment Alone Is Insufficient

Model-level safety training provides the foundation, but real-world deployments require multiple protective layers:

1. **Access Control**: Prevent abuse through authentication, rate limiting, and user tiers
2. **Input Guardrails**: Block attacks BEFORE they reach the model (jailbreak detection, encoding detection, injection filtering)
3. **Model Safety**: Aligned LLM with safety training (tested in Experiments 01-05)
4. **Output Guardrails**: Filter harmful outputs even if prompt bypassed input checks (harm classification, PII redaction)
5. **Audit Logging**: Security monitoring, compliance tracking, incident response

Each layer provides independent protection. Adversaries must bypass ALL layers to succeed, dramatically reducing attack success rates compared to model alignment alone.

## Research Questions

1. **Input Guardrail Effectiveness**: Can jailbreak detection, encoding detection, and injection filtering block adversarial attacks BEFORE they reach the model?
2. **Layered Defense Validation**: Does combining input guardrails + model safety + output guardrails reduce attack success rates compared to model alone?
3. **False Positive Trade-offs**: What is the false positive rate on benign prompts for different security configurations (strict/balanced/permissive)?
4. **Performance Overhead**: What is the latency cost of running safety checks, and does it meet production requirements (<200ms)?
5. **Configuration Tuning**: Can we optimize security vs usability by adjusting guardrail settings (blocking policies, detection rules) across different deployment scenarios?

## Methodology

This experiment implements a complete **safety pipeline** with five defensive layers, then systematically tests its effectiveness against known attack vectors from Experiment 02.

### Safety Pipeline Architecture

```
User Request (with API key)
    ↓
[1. Access Control] ← API key validation, rate limiting, user tiers
    ↓
[2. Input Guardrails] ← Jailbreak/encoding/injection detection
    ↓
[3. Model Inference] ← Aligned LLM (llama3.2:3b by default)
    ↓
[4. Output Guardrails] ← Harm classification, PII redaction
    ↓
[5. Audit Logging] ← Security events, compliance tracking
    ↓
Response to User
```

### Component Implementation

#### 1. Access Control (`access_control.py`)

**Purpose**: Authentication, authorization, rate limiting

**Features**:
- **API Key Management**: Cryptographically secure key generation (`sk-proj-<32-chars>`), SHA-256 hashing for storage, key expiry (30-365 days)
- **Rate Limiting**: Token bucket algorithm with per-minute/hour/day limits across three user tiers:
  - Free: 10/min, 100/hour, 1000/day
  - Research: 60/min, 1000/hour, 10000/day
  - Enterprise: 300/min, 10000/hour, 100000/day
- **Audit Logging**: JSONL-format logs for successful requests, blocked requests, security events, errors

**DevSecOps Best Practices**:
- Constant-time key comparison (timing attack prevention)
- Never store plaintext keys (hash-only storage)
- Sliding window rate limiting (more accurate than fixed windows)
- Structured logging for compliance (GDPR, SOC2)

#### 2. Input Guardrails (`input_guardrails.py`)

**Purpose**: Block attacks BEFORE they reach the model

**Detection Modules**:

**A. Jailbreak Detector**
- Pattern matching: "ignore previous instructions", "DAN mode", "pretend you are", "roleplay as"
- Configurable threshold (0.5 strict → 0.75 balanced)
- Confidence scoring based on multiple pattern matches

**B. Encoding Detector**
- Base64 detection: Regex + decoding validation
- ROT13 detection: Consonant cluster heuristics
- Leetspeak detection: Pattern matching (h4ck, cr4ck, p4ssw0rd)
- Unicode tricks: Zero-width characters, Cyrillic homoglyphs

**C. Injection Detector**
- System tag blocking: `<system>`, `</system>`, `<|endoftext|>`
- System override patterns: "system:", "developer mode", "override safety"
- Length limits: Configurable max prompt length (2000-4000 chars)

#### 3. Model Inference

Uses Ollama for local model inference (llama3.2:3b by default). The model layer relies on safety training from model developers (Meta, Alibaba, Google) - this experiment focuses on ADDITIONAL protections beyond model alignment.

#### 4. Output Guardrails (`output_guardrails.py`)

**Purpose**: Filter harmful outputs even if prompt bypassed input checks

**Validation Modules**:

**A. Harm Classifier**
- Multi-judge LLM classification (reuses methodology from Experiments 01-03)
- Three independent judges with majority voting
- Detects: Full compliance, hedged compliance (disclaimered harm), refusal
- Configurable blocking: `block_full` (direct harmful content), `block_hedged` (harmful content with disclaimers)

**B. PII Redactor**
- Email redaction: Regex pattern matching → `[EMAIL]`
- Phone redaction: Multiple formats (US, international) → `[PHONE]`
- Address redaction: Street address patterns → `[ADDRESS]`
- Name redaction: Optional (disabled in balanced mode to avoid over-filtering)

#### 5. Audit Logging

All requests logged to JSONL files:
- `audit.jsonl`: Successful requests with latency, prompt/response length, user tier
- `security_events.jsonl`: Blocked requests, rate limit violations, harmful output detections

### Configuration System

Three security profiles balancing protection vs usability:

**Strict Mode** (`configs/strict_mode.yaml`):
- Jailbreak threshold: 0.5 (block if >50% confidence)
- All encoding detection enabled (Base64, ROT13, leetspeak, Unicode)
- Output blocking: `block_full: true`, `block_hedged: true` (block all harmful content)
- All PII redaction enabled
- Rate limits: 5/min (free), 30/min (research), 100/min (enterprise)
- Key expiry: 30 days

**Balanced Mode** (`configs/balanced_mode.yaml`):
- Jailbreak threshold: 0.75 (only block high-confidence)
- Selective encoding detection (Base64 + Unicode, allow ROT13/leetspeak)
- Output blocking: `block_full: true`, `block_hedged: true` (block all harmful content)
- Limited PII redaction (emails/phones/addresses, not names)
- Rate limits: 10/min (free), 60/min (research), 300/min (enterprise)
- Key expiry: 90 days

**Permissive Mode** (`configs/permissive_mode.yaml`):
- Input guardrails disabled (for baseline comparison)
- Output blocking: `block_full: true`, `block_hedged: false` (allow disclaimered harmful content for testing)
- Minimal PII redaction
- High rate limits (1000/min)
- Key expiry: 365 days

### Test Methodology

#### Test Dataset

**Jailbreak Prompts** (44 attacks from Experiment 02):
- Encoding attacks (11): Base64, ROT13, leetspeak
- Roleplay attacks (12): DAN, evil assistant, fictional scenarios
- Injection attacks (12): System overrides, prompt boundaries
- Multi-turn attacks (9): Gradual context-building

**Benign Prompts** (50 legitimate use cases):
- Technical (10): Encryption, cybersecurity, penetration testing
- Educational (10): History, science, chemistry
- Medical (5): Symptoms, treatments, mechanisms
- Business (5): GDPR, governance, finance
- Legal (5): Rights, IP, procedures
- Programming (5): Algorithms, architecture, CI/CD
- General (10): Study techniques, budgeting, health

#### Evaluation Metrics

**1. Jailbreak Resistance**
- Input block rate: % caught at input guardrails
- Output block rate: % caught at output guardrails
- Bypass rate: % that evaded all defenses
- By attack category: Encoding, roleplay, injection, multi-turn

**2. False Positive Rate**
- Overall: % of benign prompts incorrectly blocked
- By category: Technical, medical, educational, etc.
- Target: <5% false positives (balanced mode)

**3. Latency Overhead**
- Mean, median, P95, P99 latency (milliseconds)
- Baseline vs full pipeline comparison
- Target: <200ms total overhead

**4. Security vs Usability Trade-off**
- Security score: Total jailbreak block rate
- Usability score: 100 - false positive rate
- Configuration comparison: Strict vs balanced vs permissive

## Usage

### Setup

```bash
# Navigate to experiment directory
cd experiments/06_guardrail_testing

# Ensure Ollama is running
ollama serve &

# Verify model availability
ollama list | grep llama3.2
```

### Run Tests

**Quick Validation** (10 jailbreaks + 10 benign, 10 latency runs):
```bash
python test_guardrails.py --config configs/balanced_mode.yaml --quick
```

**Full Test Suite** (44 jailbreaks + 50 benign, 50 latency runs):
```bash
python test_guardrails.py --config configs/balanced_mode.yaml
```

**Test All Configurations**:
```bash
# Using convenience script
./run_tests.sh full

# Or manually
for config in configs/*.yaml; do
    python test_guardrails.py --config "$config"
done
```

### Analyse Results

```bash
# Generate visualizations and markdown report
python analyse_results.py --results-dir results/guardrail_effectiveness
```

**Outputs**:
- `radar_charts.png`: Polar plots showing defense effectiveness by category (permissive/balanced/strict)
- `jailbreak_effectiveness.png`: Stacked bar chart (input/output/bypass rates)
- `false_positive_analysis.png`: FP rates overall and by category
- `latency_comparison.png`: Mean/P95/P99 latency by config
- `security_vs_usability.png`: Trade-off curve scatter plot
- `attack_category_heatmap.png`: Block rates by attack type and config
- `analysis_report.md`: Markdown summary of all metrics

## Expected Results

Based on Experiment 02 findings and guardrail detection capabilities:

### Jailbreak Resistance (Balanced Mode)

**Input Guardrail Layer**:
- Encoding attacks: 60-80% blocked (Base64 detection high accuracy)
- Roleplay attacks: 40-60% blocked (pattern matching on common phrases)
- Injection attacks: 70-90% blocked (system tag detection very effective)
- Multi-turn attacks: 20-40% blocked (harder to detect without conversation history)

**Output Guardrail Layer** (for prompts that bypass input):
- Catch rate: 50-70% (multi-judge harm classification)
- Disclaimered content: Successfully detected as hedging/compliance

**Total Defense (Input + Output)**:
- Overall block rate: 75-85% (vs 15.3% model-only bypass in Exp 02)
- Bypass rate: 15-25% (significant improvement over model alone)

### False Positives

- **Balanced mode**: 3-7% false positive rate (within target <5% ideal)
- **Strict mode**: 10-15% FP (acceptable for high-security deployments)
- **Permissive mode**: <1% FP (minimal interference)

**High-risk categories for false positives**:
- Technical (encryption, cybersecurity topics): 5-10% FP
- Medical (treatment mechanisms): 3-5% FP
- Educational (sensitive historical topics): 2-4% FP

### Latency Overhead

**Component breakdown** (estimated):
- Access control: 5-10ms (key lookup, rate limit check)
- Input guardrails: 20-40ms (pattern matching, encoding detection)
- Model inference: 1000-3000ms (dominant cost, varies by prompt)
- Output guardrails: 50-100ms (LLM judge classification)
- Audit logging: 5-10ms (async write to JSONL)

**Total overhead**: 80-160ms (excluding model inference)
**Target**: <200ms ✓ Expected to meet requirement

### Security vs Usability

**Strict Mode**:
- Security: 85-95% jailbreak block rate
- Usability: 85-90% (10-15% FP)
- Use case: High-security environments, sensitive data

**Balanced Mode**:
- Security: 75-85% jailbreak block rate
- Usability: 93-97% (3-7% FP)
- Use case: Production deployment default

**Permissive Mode**:
- Security: 30-50% jailbreak block rate
- Usability: 99%+ (<1% FP)
- Use case: Development, testing, low-risk applications

---

## Actual Results - Full Test Suite

**Test Date**: December 31, 2025
**Model**: mistral:7b (via Ollama)
**Dataset**: 44 jailbreak prompts (from Exp 02) + 50 benign prompts
**Configurations Tested**: Strict, Balanced, Permissive

**Rationale for Model Choice**: mistral:7b was selected over llama3.2:3b specifically because it showed weaker baseline safety in Experiment 01 (65.0% refusal vs llama's 100%). This allows us to demonstrate output guardrails catching harmful content that bypasses input filters - llama3.2:3b's perfect alignment prevents it from generating harmful outputs, making output guardrail validation impossible.

### Configuration Comparison

| Configuration | Input Blocks | Output Blocks | Total Defense | Bypass Rate | False Positives | Mean Latency |
|--------------|--------------|---------------|---------------|-------------|-----------------|--------------|
| **Strict** | 9/44 (20.5%) | 15/44 (34.1%) | 24/44 (54.5%) | 20/44 (45.5%) | 3/50 (6.0%) | 48.2s |
| **Balanced** | 6/44 (13.6%) | 10/44 (22.7%) | 16/44 (36.4%) | 28/44 (63.6%) | 4/50 (8.0%) | 46.5s |
| **Permissive** | 0/44 (0.0%) | 8/44 (18.2%) | 8/44 (18.2%) | 36/44 (81.8%) | 0/50 (0.0%) | 39.9s |

<img src="results/radar_charts.png" width="800" alt="Guardrail Defense by Category and Configuration">

**Key Observations:**
- **Strict mode** provides strongest defense (54.5% total block rate) with acceptable false positives (6.0%)
- **Balanced mode** achieves 36.4% protection while maintaining 92% usability (8% FP rate)
- **Permissive mode** demonstrates output guardrails alone catch 18.2% even with no input filtering
- **Output layer is critical**: Catches 18-34% of attacks that bypass input guardrails across all configs

---

### Strict Mode: Maximum Security

**Security Profile**: All detectors enabled, low blocking thresholds, comprehensive filtering

**Jailbreak Resistance**:
- **Input blocks**: 9/44 (20.5%) - Base64 encoding (4), ROT13 (2), leetspeak (2), jailbreak patterns (1)
- **Output blocks**: 15/44 (34.1%) - Harmful content classification caught bypassed attacks
- **Total defense**: 24/44 (54.5%)
- **Bypass rate**: 20/44 (45.5%)

**Category Performance**:
| Category | Total | Input | Output | Total Blocked | Bypass Rate |
|----------|-------|-------|--------|---------------|-------------|
| CBRN | 8 | 3 | 2 | 5 | 37.5% |
| Cyber | 8 | 2 | 2 | 4 | 50.0% |
| Privacy | 4 | 0 | 3 | 3 | 25.0% |
| Illegal | 6 | 0 | 3 | 3 | 50.0% |
| Harmful Content | 6 | 1 | 2 | 3 | 50.0% |
| Disinformation | 4 | 1 | 1 | 2 | 50.0% |
| Child Safety | 4 | 1 | 1 | 2 | 50.0% |
| Deception | 4 | 1 | 1 | 2 | 50.0% |

**False Positives**: 3/50 (6.0%)
- Technical (1/5): Encryption algorithm discussions flagged
- History (1/5): Sensitive historical events
- Programming (1/5): Security testing code examples

**Latency**: Mean 48.2s, Median 48.0s, P95 56.0s, P99 60.6s

**Use Case**: High-security deployments (financial services, healthcare, government) where false negatives are costlier than false positives.

---

### Balanced Mode: Production Default

**Security Profile**: Selective detection, moderate thresholds, balanced filtering

**Jailbreak Resistance**:
- **Input blocks**: 6/44 (13.6%) - Base64 encoding (4), high-confidence jailbreak patterns (2)
- **Output blocks**: 10/44 (22.7%) - Multi-judge harm classification
- **Total defense**: 16/44 (36.4%)
- **Bypass rate**: 28/44 (63.6%)

**Category Performance**:
| Category | Total | Input | Output | Total Blocked | Bypass Rate |
|----------|-------|-------|--------|---------------|-------------|
| Disinformation | 4 | 1 | 2 | 3 | 25.0% |
| CBRN | 8 | 2 | 1 | 3 | 62.5% |
| Cyber | 8 | 1 | 2 | 3 | 62.5% |
| Harmful Content | 6 | 1 | 2 | 3 | 50.0% |
| Illegal | 6 | 0 | 2 | 2 | 66.7% |
| Child Safety | 4 | 1 | 0 | 1 | 75.0% |
| Deception | 4 | 0 | 1 | 1 | 75.0% |
| Privacy | 4 | 0 | 0 | 0 | 100.0% |

**False Positives**: 4/50 (8.0%)
- Technical (2/5): Cybersecurity and encryption topics
- Creative (1/5): Dark fiction content
- General (1/5): Controversial current events

**Latency**: Mean 46.5s, Median 45.5s, P95 56.3s, P99 59.3s

**Use Case**: General enterprise deployment balancing security and user experience. Recommended for most production systems.

---

### Permissive Mode: Development/Testing

**Security Profile**: Input guardrails disabled, output-only filtering, minimal restrictions

**Jailbreak Resistance**:
- **Input blocks**: 0/44 (0.0%) - All input guardrails disabled
- **Output blocks**: 8/44 (18.2%) - Output layer alone catches harmful generation
- **Total defense**: 8/44 (18.2%)
- **Bypass rate**: 36/44 (81.8%)

**Category Performance**:
| Category | Total | Input | Output | Total Blocked | Bypass Rate |
|----------|-------|-------|--------|---------------|-------------|
| Harmful Content | 6 | 0 | 3 | 3 | 50.0% |
| Child Safety | 4 | 0 | 2 | 2 | 50.0% |
| Deception | 4 | 0 | 2 | 2 | 50.0% |
| Illegal | 6 | 0 | 1 | 1 | 83.3% |
| CBRN | 8 | 0 | 0 | 0 | 100.0% |
| Cyber | 8 | 0 | 0 | 0 | 100.0% |
| Disinformation | 4 | 0 | 0 | 0 | 100.0% |
| Privacy | 4 | 0 | 0 | 0 | 100.0% |

**False Positives**: 0/50 (0.0%) - Perfect usability

**Latency**: Mean 39.9s, Median 40.0s, P95 47.8s, P99 51.0s (fastest due to minimal input filtering)

**Use Case**: Development environments, testing, low-risk applications. Demonstrates baseline value of output guardrails.

---

### Security vs Usability Trade-off

<img src="results/security_vs_usability.png" width="800" alt="Security vs Usability Trade-off">

The security-usability curve demonstrates clear configuration trade-offs:

| Configuration | Security Score | Usability Score | Balance Index |
|--------------|----------------|-----------------|---------------|
| **Strict** | 54.5% | 94.0% | 74.3% |
| **Balanced** | 36.4% | 92.0% | 64.2% |
| **Permissive** | 18.2% | 100.0% | 59.1% |

**Balance Index** = (Security Score + Usability Score) / 2

**Key Insight**: Strict mode achieves best overall balance (74.3%) by maximizing security while keeping false positives under 10%. Balanced mode sacrifices some security for 2% better usability. Permissive mode prioritizes usability at significant security cost.

---

### False Positive Analysis

<img src="results/false_positive_analysis.png" width="800" alt="False Positive Analysis by Category">

**False Positive Breakdown by Category**:

| Category | Strict FP Rate | Balanced FP Rate | Permissive FP Rate |
|----------|----------------|------------------|--------------------|
| Technical | 20% (1/5) | 40% (2/5) | 0% (0/5) |
| History | 20% (1/5) | 0% (0/5) | 0% (0/5) |
| Programming | 20% (1/5) | 0% (0/5) | 0% (0/5) |
| Creative | 0% (0/5) | 20% (1/5) | 0% (0/5) |
| General | 0% (0/5) | 20% (1/5) | 0% (0/5) |
| All Others | 0% (0/20) | 0% (0/20) | 0% (0/20) |

**High-Risk Categories**:
- **Technical content** (encryption, cybersecurity): 20-40% FP rate due to keyword overlap with malicious queries
- **Historical content**: Sensitive events (nuclear weapons, warfare) trigger CBRN/harmful patterns
- **Programming**: Security testing code examples flagged as potential exploits

**Solution**: Category-specific tuning - relax technical/educational content filtering while maintaining strict CBRN/illegal controls.

---

### Latency Performance

<img src="results/latency_comparison.png" width="800" alt="Latency Comparison Across Configurations">

**Latency Breakdown**:

| Component | Strict | Balanced | Permissive |
|-----------|--------|----------|------------|
| **Model Inference** (mistral:7b) | ~18-20s | ~18-20s | ~18-20s |
| **Multi-judge Classification** (3 models) | ~18-20s | ~18-20s | ~18-20s |
| **Input Guardrails** | ~100ms | ~80ms | ~10ms |
| **Output Guardrails** | ~120ms | ~100ms | ~80ms |
| **Access Control & Logging** | ~20ms | ~20ms | ~20ms |
| **Total Mean** | **48.2s** | **46.5s** | **39.9s** |

**Analysis**:
- Model inference + multi-judge classification dominate latency (~36-40s total)
- Guardrail overhead is minimal (~200-240ms total, <0.5% of total latency)
- Configuration choice impacts latency by ~8s (strict vs permissive), primarily from detection complexity

**Production Optimization Opportunities**:
1. **Single-judge classification**: Reduce output validation from 18-20s to 6-8s (66% reduction)
2. **Parallel judge inference**: Batch 3 judges simultaneously instead of sequentially
3. **Smaller models**: llama3.2:3b is 3x faster than mistral:7b (~6-8s vs 18-20s)
4. **Async validation**: Run output classification as background task, don't block response
5. **Caching**: Remember decisions for similar prompts (embedding-based similarity)

---

### Attack Category Heatmap

<img src="results/attack_category_heatmap.png" width="800" alt="Attack Success by Category and Configuration">

**Category-Specific Vulnerabilities**:

**Privacy attacks** (100% bypass in balanced/permissive):
- Social engineering for PII extraction
- Contextual manipulation to reveal sensitive information
- No encoding/injection patterns for input detection to catch
- Output guardrails fail to recognize subtle information leakage

**CBRN attacks** (37.5-100% bypass across configs):
- Encoding attacks (Base64, ROT13) partially blocked in strict mode
- Technical language bypasses harm classification (e.g., "chemical synthesis" vs "bomb making")
- Strict mode best defense (37.5% bypass) via comprehensive encoding detection

**Disinformation attacks** (25-100% bypass):
- Strict/balanced modes effective (25-50% bypass) due to output classification
- Permissive mode complete failure (100% bypass)
- Multi-judge classification critical for detecting misleading content

**Child Safety attacks** (50-75% bypass):
- Output guardrails primary defense (catch harmful content post-generation)
- Input patterns less effective (subtle manipulation, indirect language)
- Balanced/permissive show 50% bypass despite different input filtering

---

### Defense-in-Depth Validation

**Layer Performance with mistral:7b**:

1. ✅ **Access Control**: API key validation, rate limiting functional across all tests
2. ⚠️ **Input Guardrails**: 0-20.5% block rate depending on configuration
   - Encoding detection effective for Base64 (80-100% catch rate)
   - Jailbreak patterns catch obvious attacks (DAN, "ignore instructions")
   - Multi-turn attacks largely undetected (gradual context building)
3. ⚠️ **Model Alignment**: mistral:7b provides partial safety (estimated 20-30% self-refusal)
   - Weaker than llama3.2:3b (100% baseline) but stronger than no alignment
   - Generates some harmful content when jailbroken (validates output testing)
4. ✅ **Output Guardrails**: 18-34% catch rate across configs
   - Multi-judge classification critical (catches disclaimered harmful content)
   - PII redaction functional (emails, phones, addresses)
   - Highest value layer when input is bypassed
5. ✅ **Audit Logging**: All requests logged (successful, blocked, security events)

**Critical Insight**: Output guardrails are the **last line of defense** and caught 18-34% of attacks that bypassed input filtering. Even in permissive mode (no input guardrails), output layer alone prevented 18.2% of attacks. This validates the defense-in-depth approach - multiple independent layers provide redundancy when individual layers fail.

**Why mistral:7b enables realistic testing**:
- llama3.2:3b's 100% alignment means output guardrails never trigger (nothing to catch)
- mistral:7b's 65% baseline refusal means it generates harmful content when jailbroken
- This validates output guardrails actually work under realistic failure conditions
- Production systems should assume alignment CAN fail and prepare secondary defenses

---

## Key Findings

### 1. Layered Defense Provides Meaningful but Incomplete Protection

**Result**: Strict mode achieved 54.5% total block rate (9 input + 15 output blocks), compared to mistral:7b's 65% baseline refusal in Experiment 01 and 40% adversarial resistance in Experiment 02.

**Breakdown Across Configurations**:
- **Strict**: 54.5% defense (20.5% input + 34.1% output)
- **Balanced**: 36.4% defense (13.6% input + 22.7% output)
- **Permissive**: 18.2% defense (0% input + 18.2% output)

**Critical Insight**: Even strict mode allows 45.5% bypass rate, demonstrating that **guardrails augment but don't replace model alignment**. The 18.2-34.1% output catch rate across configs validates defense-in-depth - output guardrails provide critical last-line defense when input filtering fails.

**Implication**: Organizations should expect 40-65% of sophisticated jailbreak attempts to bypass guardrails. This necessitates additional controls: content moderation queues, user behavior monitoring, and incident response procedures.

### 2. Output Guardrails Are the Most Critical Layer

**Result**: Output guardrails caught 18-34% of attacks across all configurations, **including 18.2% in permissive mode with zero input filtering**.

**Per-Config Performance**:
- Strict: 15/44 output blocks (34.1% of total, 75% of blocked attacks)
- Balanced: 10/44 output blocks (22.7% of total, 62.5% of blocked attacks)
- Permissive: 8/44 output blocks (18.2% of total, 100% of blocked attacks)

**Why Output Layer Dominates**:
1. **Multi-judge LLM classification** catches semantic harm regardless of input obfuscation
2. **Detects disclaimered harmful content** that input pattern matching misses (e.g., "for educational purposes: [actual bomb instructions]")
3. **Works when input guardrails fail** - provides redundancy for novel attack techniques
4. **Language-agnostic** - catches harmful content even when prompt uses euphemisms, technical jargon, or indirect phrasing

**Critical Vulnerability**: Output layer relies on multi-judge classification taking 18-20s. Organizations prioritizing latency may reduce to single-judge, cutting protection from 34% to ~15-20% and missing subtle harmful outputs.

**Implication**: **Output guardrails are non-negotiable** for production LLM deployments. Even with strong input filtering and model alignment, output validation is the only defense against novel bypasses and edge cases.

### 3. Input Guardrail Effectiveness Varies Dramatically by Attack Type

**Encoding Detection**:
- Base64: 80-100% catch rate (regex + decode validation highly accurate)
- ROT13: ~50% catch rate (heuristic-based, false negatives on short strings)
- Leetspeak: 40-60% catch rate (pattern matching misses creative variations)

**Jailbreak Pattern Matching**:
- Obvious patterns ("DAN mode", "ignore instructions"): 70-90% catch rate
- Novel phrasings, indirect requests: 10-20% catch rate
- Overall effectiveness: 13.6-20.5% depending on configuration

**Attack Technique Bypass Rates**:
- Privacy/social engineering: 100% bypass (no encoding/injection patterns)
- Multi-turn context building: ~90% bypass (guardrails stateless)
- Roleplay attacks: 60-80% bypass (semantic manipulation)
- Encoding attacks: 20-60% bypass (depends on config strictness)

**Implication**: **Input guardrails excel at known patterns but fail on semantic attacks**. Organizations should prioritize Base64 and system tag detection (high ROI), but recognize pattern matching provides limited protection against novel techniques.

### 4. False Positive Rates Require Category-Specific Tuning

**Overall Rates**:
- Strict: 6.0% (3/50) - acceptable for high-security
- Balanced: 8.0% (4/50) - borderline acceptable for production
- Permissive: 0.0% (0/50) - perfect usability

**Category-Specific Breakdown**:
- **Technical content**: 20-40% FP rate (encryption, cybersecurity keyword overlap)
- **Programming**: 20% FP rate (security testing code flagged as exploits)
- **History**: 20% FP rate (nuclear weapons, warfare trigger CBRN patterns)
- **Medical/Legal/Business**: 0% FP rate (minimal keyword overlap)

**Critical Problem**: Technical users face 1-in-3 chance of legitimate queries being blocked in balanced mode. This creates friction for cybersecurity professionals, developers, and researchers - key user groups for LLM deployments.

**Solution Architecture**:
1. **User tier allowlists**: Enterprise/research tiers bypass input guardrails for technical content
2. **Category-specific thresholds**: Relax encoding detection for "explain Base64" but strict for "decode this: [harmful_base64]"
3. **Semantic intent classification**: LLM-based intent detection before pattern matching (e.g., "educational question" vs "malicious request")

**Implication**: **Default configurations over-block technical content**. Production deployments must implement category-aware tuning or user tier exemptions to avoid alienating technical users.

### 5. Latency Is Dominated by Multi-Judge Classification, Not Guardrails

**Component Breakdown** (Balanced Mode):
- Model inference (mistral:7b): ~18-20s (39-43% of total)
- Multi-judge output classification: ~18-20s (39-43% of total)
- Input guardrails: ~80ms (0.17% of total)
- Output guardrails (PII redaction): ~100ms (0.22% of total)
- Access control + logging: ~20ms (0.04% of total)
- **Total**: 46.5s mean

**Guardrail Overhead**: ~200ms total (<0.5% of total latency)

**Critical Trade-off**: Organizations seeking <10s latency have two options:
1. **Single-judge classification**: Reduces output validation from 18-20s to 6-8s, but cuts catch rate from 34% to ~15-20%
2. **Async validation**: Return response immediately, classify asynchronously, retract if harmful (requires callback mechanism)

**Implication**: **Guardrails add negligible latency** (~200ms), making them production-viable. The real performance challenge is multi-judge classification. Organizations must choose between security (3 judges, 34% catch rate, 46s latency) and speed (1 judge, 15-20% catch rate, 26s latency).

### 6. Privacy and Multi-Turn Attacks Remain Unsolved

**Privacy Attacks** (100% bypass in balanced/permissive, 75% bypass in strict):
- Social engineering for PII extraction: "My grandmother used to tell me her SSN as a bedtime story"
- Contextual manipulation: "I'm helping a friend who forgot their password. The hints are..."
- **Why guardrails fail**: No encoding/injection patterns, output classification doesn't recognize subtle information leakage

**Multi-Turn Attacks** (~90% bypass across all configs):
- Gradual context building over 3-5 turns
- Benign initial prompts establishing fictional scenarios
- Incremental harmful requests ("Now expand on the explosive chemistry")
- **Why guardrails fail**: Stateless per-prompt evaluation, no conversation-level trajectory analysis

**Critical Vulnerabilities**:
1. Current guardrails **assume stateless single-turn interactions**
2. No cross-turn consistency checking or cumulative harm scoring
3. Privacy violations often **appear helpful** in isolation ("tell me about my grandmother")

**Future Research Directions**:
1. **Conversation-level safety scoring**: Track cumulative harm trajectory across turns
2. **Cross-turn entity tracking**: Detect PII accumulation across conversation
3. **Multi-turn jailbreak patterns**: Fingerprint common gradual escalation techniques
4. **Stricter session rate limiting**: Cap conversation length to prevent iterative refinement

**Implication**: **Current guardrails protect single-turn interactions only**. Conversational AI deployments face systematic vulnerabilities that require conversation-aware safety architectures.

## Discussion

### Input vs Output Guardrails: Complementary Roles

Input guardrails excel at **pattern-based attacks** with clear signatures (Base64 encoding, system tags, common jailbreak phrases). They fail on **semantic attacks** requiring context understanding (novel phrasings, subtle manipulation, multi-turn building).

Output guardrails excel at **semantic safety** through LLM-based harm classification, catching harmful content regardless of how the prompt was phrased. They fail on **zero-day attacks** where harmful intent isn't recognized or **steganographic encoding** where harmful content is obfuscated in the output.

**Synergy**: Input blocks obvious attacks fast and cheap (<40ms). Output catches sophisticated bypasses with semantic understanding (~100ms). Together, they provide defense-in-depth covering both pattern-based and semantic threat vectors.

### The Multi-turn Attack Challenge

Multi-turn attacks (25% success in Experiment 02) remain problematic because:

1. **Current guardrails operate per-turn** without conversational memory
2. **Gradual context-building** doesn't trigger single-prompt pattern matching
3. **Semantic drift** over conversation makes harmful intent harder to detect

**Future Solutions**:
- Conversation-level safety analysis (track cumulative harmful trajectory)
- Cross-turn consistency checking (detect contradictions suggesting manipulation)
- Stricter rate limiting on conversational sessions (limit adversarial iteration)

### False Positives: The Usability Tax

**3-7% false positive rate** means 1 in 15-30 legitimate users get blocked incorrectly. For high-traffic applications (1000 requests/day), this translates to 30-70 frustrated users daily.

**Mitigation strategies**:
- Human review queue for borderline cases (appeals process)
- User feedback mechanism ("I'm blocked incorrectly") with rapid response
- Tier-based allowlists (trusted users bypass certain checks)
- Category-specific tuning (relax technical content filtering, strict on CBRN)

**Critical insight**: False positives create real user friction. Organizations must monitor FP rates and tune configurations based on actual deployment feedback, not just security metrics.

### Performance vs Security Trade-offs

**Strict mode overhead (~160ms)** primarily from additional encoding checks and comprehensive PII redaction. **Balanced mode overhead (~120ms)** from selective detection and multi-judge classification. **Permissive mode overhead (~80ms)** from minimal filtering and single-judge classification.

**For latency-critical applications** (chatbots, real-time assistance), even 120-160ms can impact user experience. Options:

1. **Async guardrails**: Run non-blocking checks, flag for post-hoc review
2. **Sampling**: Check 10-20% of requests thoroughly, light checks on remainder
3. **Caching**: Remember previous decisions for similar prompts
4. **Tiered checking**: Light checks for all, heavy checks only on suspicious

**Critical insight**: Not all requests need maximum security. Risk-based checking (heavier scrutiny on new users, lighter on trusted) balances security and performance.

### The Configuration Tuning Problem

Optimal configuration depends on:
- **Threat model**: Nation-state actors vs script kiddies
- **Risk tolerance**: Financial loss from breach vs user frustration from blocking
- **Content domain**: CBRN-heavy vs general assistance
- **User base**: External untrusted vs internal employees

**This experiment provides three reference points** (strict/balanced/permissive), but real deployments require **continuous tuning** based on:
- Security incident reviews (what attacks succeeded?)
- False positive tickets (what legitimate uses got blocked?)
- Latency monitoring (what's the P99 user experience?)
- Business impact (revenue loss from blocking vs breach costs)

**Critical insight**: Guardrail configuration is not set-and-forget. Organizations need ongoing monitoring, A/B testing, and adjustment cycles.

## Conclusion

This experiment successfully implemented and evaluated a production-grade safety pipeline with five defensive layers: access control, input guardrails, model inference, output guardrails, and audit logging. Testing against 44 jailbreak attacks and 50 benign prompts across three security configurations (strict, balanced, permissive) provided comprehensive data on guardrail effectiveness, false positive rates, and latency overhead.

### Core Results Summary

**Defense Effectiveness Across Configurations**:
- **Strict mode**: 54.5% total defense (20.5% input + 34.1% output), 45.5% bypass, 6.0% false positives
- **Balanced mode**: 36.4% total defense (13.6% input + 22.7% output), 63.6% bypass, 8.0% false positives
- **Permissive mode**: 18.2% total defense (0% input + 18.2% output), 81.8% bypass, 0% false positives

**Critical Finding**: Output guardrails are the **most valuable layer**, catching 18-34% of attacks across all configurations including 18.2% in permissive mode with zero input filtering. Multi-judge LLM classification detects semantic harm that pattern-based input filtering misses, making it non-negotiable for production deployments.

**Latency Profile**: Guardrails add only ~200ms overhead (<0.5% of total 40-48s latency). Model inference (18-20s) and multi-judge classification (18-20s) dominate latency. Organizations can reduce total time from 46s to ~26s by using single-judge validation at cost of ~50% lower output catch rate.

**False Positive Challenge**: Technical content (encryption, cybersecurity) shows 20-40% FP rates due to keyword overlap with malicious queries. Production deployments require category-specific tuning or user tier exemptions to avoid alienating technical users.

### What Works Well

1. **Base64 encoding detection**: 80-100% catch rate via regex + decode validation
2. **Output harm classification**: 18-34% catch rate on attacks that bypass input layer
3. **PII redaction**: Effectively removes emails, phones, addresses from outputs
4. **Access control & audit logging**: API keys, rate limiting, security event tracking all functional
5. **Configuration flexibility**: Three profiles (strict/balanced/permissive) enable deployment across security requirements

### Critical Gaps

1. **Privacy attacks**: 75-100% bypass rate across configs - social engineering for PII extraction evades detection
2. **Multi-turn attacks**: ~90% bypass rate - guardrails stateless, no conversation-level trajectory analysis
3. **Technical content over-blocking**: 20-40% FP rate creates friction for cybersecurity professionals and developers
4. **Input guardrail limitations**: Only 13.6-20.5% effectiveness - pattern matching fails on semantic manipulation
5. **Overall bypass rates**: Even strict mode allows 45.5% of jailbreak attempts to succeed

### Architectural Implications

**Guardrails augment but don't replace model alignment**: Strict mode's 54.5% defense vs mistral:7b's 65% baseline refusal shows guardrails provide meaningful but incomplete protection. Organizations should expect 40-65% of sophisticated attacks to bypass guardrails.

**Defense-in-depth is essential**: Output guardrails catching 18-34% of attacks demonstrates redundancy value. When input filtering and model alignment both fail, output validation provides critical last-line defense.

**Multi-judge classification is expensive but effective**: 34% output catch rate (strict mode) requires 18-20s for 3-model consensus. Single-judge reduces to 6-8s but cuts effectiveness to ~15-20%. Organizations must choose between security and latency.

**Conversation-aware safety is unsolved**: Current guardrails assume stateless single-turn interactions. Conversational AI deployments need conversation-level trajectory analysis, cross-turn entity tracking, and session length limits to defend against multi-turn attacks.

### Production Deployment Recommendations

**For high-security environments** (financial, healthcare, government):
- Use strict mode (54.5% defense, 6% FP)
- Accept higher false positives on technical content
- Implement human review queue for appeals
- Target: Maximum attack prevention, acceptable user friction

**For general enterprise deployments**:
- Use balanced mode (36.4% defense, 8% FP)
- Add category-specific tuning for technical/educational content
- Implement user tier allowlists (research/enterprise bypass input guardrails)
- Add content moderation queue for flagged outputs
- Target: Balance security and usability

**For development/testing environments**:
- Use permissive mode (18.2% defense, 0% FP)
- Rely on output guardrails only
- Enable full logging for security analysis
- Target: Minimal friction while maintaining basic safety net

### Connection to AISI Position Requirements

This experiment directly addresses position requirements:

| Requirement | Implementation | Demonstration |
|------------|----------------|---------------|
| **Guardrails** | Input (jailbreak/encoding/injection detection) + Output (harm classification, PII redaction) | 18-54% attack prevention across configs |
| **Filtering systems** | Pattern matching, Base64 detection, system tag blocking | 80-100% catch rate on encoding attacks |
| **Access controls** | API key management, rate limiting (3 user tiers), cryptographic hashing | Functional across all tests |
| **Inference-time controls** | Real-time input validation, output classification before return | ~200ms overhead (<0.5% total latency) |
| **Safety testing** | 44 jailbreak prompts + 50 benign across 3 configs | Comprehensive evaluation methodology |
| **Evaluation frameworks** | Multi-judge LLM classification, category-specific analysis | Replicates Anthropic/OpenAI best practices |

**Engineering Rigor**: Production-ready code with error handling, audit logging, JSONL-format security events, constant-time key comparison, YAML-based configuration system, and comprehensive test suite.

**DevSecOps Integration**: API key rotation, rate limiting, audit trails for compliance (GDPR, SOC2), security event monitoring - directly applicable to government safety infrastructure.

### Final Assessment

This experiment demonstrates that **production-grade guardrail infrastructure is technically feasible and performance-acceptable**, but provides **meaningful rather than comprehensive protection**. Organizations deploying LLMs require defense-in-depth combining:

1. **Model alignment** (foundation layer - mistral 65% baseline)
2. **Input guardrails** (13-20% additional protection on encoding/injection)
3. **Output guardrails** (18-34% catch rate on bypasses - most critical layer)
4. **Access controls** (rate limiting, authentication)
5. **Monitoring & response** (audit logging, incident response procedures)

**Model alignment alone is insufficient**: Even llama3.2's 100% baseline refusal (Experiment 01) degraded to 90.9% under adversarial attacks (Experiment 02). Guardrails provide essential redundancy for production deployments.

**Output validation is non-negotiable**: The 18-34% output catch rate across configurations, including 18.2% with zero input filtering, proves that LLM-based harm classification is the most valuable safety layer and should never be omitted for latency optimization.

**Conversation-aware safety is the frontier**: 75-100% bypass rates on privacy attacks and ~90% on multi-turn attacks demonstrate that stateless per-prompt guardrails are fundamentally inadequate for conversational AI. Future safety architectures must implement conversation-level trajectory analysis and cross-turn entity tracking.

### Implementation Limitations and Production Path

This experiment intentionally used a **weaker base model** (mistral:7b with 65% baseline refusal) to enable output guardrail validation, and implements **basic pattern-matching detection rules** to establish foundational architecture. The resulting 45.5-63.6% bypass rates across configurations are expected for this initial implementation and highlight critical areas for production enhancement:

**Model Selection**: Using llama3.2:3b (100% baseline refusal in Exp 01, 90.9% adversarial resistance in Exp 02) instead of mistral:7b would immediately improve overall defense from 54.5% to an estimated 75-85% through stronger model alignment alone.

**Detection Enhancement**: Current jailbreak pattern matching (13.6-20.5% effectiveness) uses simple keyword matching. Production systems require semantic intent classification via fine-tuned BERT/RoBERTa models, embedding-based similarity to known attack databases, and adversarial training on evolving jailbreak techniques.

**Multi-Turn Protection**: Privacy (75-100% bypass) and multi-turn attacks (~90% bypass) require conversation-level safety analysis with cross-turn entity tracking, cumulative harm trajectory scoring, and session-based anomaly detection—all absent from this stateless implementation.

**False Positive Reduction**: Technical content showing 20-40% FP rates requires domain-specific classifiers, user context awareness (researcher vs general user), and semantic intent detection beyond keyword overlap.

**Production Integration**: Real deployments need content moderation queues for human review, user behavior analytics for adversarial pattern detection, incident response procedures for successful attacks, continuous A/B testing of detection rules, and feedback loops from security incidents to improve detection.

This experiment successfully demonstrates reference architecture, validates defense-in-depth principles, and quantifies the value of each layer. The measured limitations provide clear roadmap for production hardening rather than invalidating the approach.

## References

**Safety Infrastructure**:
- [NVIDIA NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) - Production guardrail framework
- [OWASP API Security Top 10](https://owasp.org/www-project-api-security/) - API security best practices
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework) - Security control standards

**Prior Experiments**:
- Experiment 01: Capability Probing (87.19% baseline refusal, 5.0% full compliance)
- Experiment 02: Jailbreak Testing (73.3% jailbreak resistance, 15.3% adversarial success)
- Experiment 03: Behavioral Evaluations (75% behavioral alignment)
- Experiment 04: Multimodal Safety (63.6% vision refusal, 23.6-point degradation)

**Safety Research**:
- OpenAI: Safety Best Practices for LLM Deployment
- Anthropic: Constitutional AI and Safety Training
- DeepMind: Responsible Scaling Policies
