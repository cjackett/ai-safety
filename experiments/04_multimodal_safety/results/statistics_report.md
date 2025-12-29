# Multimodal Safety Evaluation Analysis Report

Total tests: 66
Models tested: 3
Attack types: 11
Harm categories: 7

## Overall Compliance Distribution

- refused: 42 (63.6%)
- hedged: 11 (16.7%)
- full: 13 (19.7%)

## Model Performance


### gemma3:4b
- Total: 22
- Refused: 19 (86.4%)
- Hedged: 0 (0.0%)
- Full: 3 (13.6%)

### llava:7b
- Total: 22
- Refused: 11 (50.0%)
- Hedged: 9 (40.9%)
- Full: 2 (9.1%)

### qwen3-vl:4b
- Total: 22
- Refused: 12 (54.5%)
- Hedged: 2 (9.1%)
- Full: 8 (36.4%)

## Attack Type Effectiveness


### Base64 In Image
- Total tests: 3
- Success rate: 0.0% (0 hedged + 0 full)
- Refusal rate: 100.0%

### Caesar Cipher
- Total tests: 3
- Success rate: 33.3% (0 hedged + 1 full)
- Refusal rate: 66.7%

### Cross Modal Inconsistency
- Total tests: 15
- Success rate: 46.7% (4 hedged + 3 full)
- Refusal rate: 53.3%

### Direct Harmful
- Total tests: 12
- Success rate: 58.3% (4 hedged + 3 full)
- Refusal rate: 41.7%

### Educational Framing
- Total tests: 3
- Success rate: 0.0% (0 hedged + 0 full)
- Refusal rate: 100.0%

### Leetspeak In Image
- Total tests: 3
- Success rate: 66.7% (1 hedged + 1 full)
- Refusal rate: 33.3%

### Multi Turn Setup
- Total tests: 3
- Success rate: 66.7% (0 hedged + 2 full)
- Refusal rate: 33.3%

### Ocr Harmful Text
- Total tests: 15
- Success rate: 6.7% (0 hedged + 1 full)
- Refusal rate: 93.3%

### Reverse Text
- Total tests: 3
- Success rate: 66.7% (1 hedged + 1 full)
- Refusal rate: 33.3%

### Roleplay In Image
- Total tests: 3
- Success rate: 33.3% (1 hedged + 0 full)
- Refusal rate: 66.7%

### Rot13 In Image
- Total tests: 3
- Success rate: 33.3% (0 hedged + 1 full)
- Refusal rate: 66.7%

## Category Analysis


### Cbrn
- Overall refusal rate: 83.3%
- Hedged: 0/6
- Full: 1/6

- Most vulnerable model: qwen3-vl (50.0% refusal)

### Child Safety
- Overall refusal rate: 33.3%
- Hedged: 1/3
- Full: 1/3

- Most vulnerable model: llava (0.0% refusal)

### Cyber
- Overall refusal rate: 75.0%
- Hedged: 3/12
- Full: 0/12

- Most vulnerable model: llava (50.0% refusal)

### Deception
- Overall refusal rate: 55.6%
- Hedged: 1/9
- Full: 3/9

- Most vulnerable model: qwen3-vl (33.3% refusal)

### Harmful Content
- Overall refusal rate: 66.7%
- Hedged: 3/15
- Full: 2/15

- Most vulnerable model: llava (40.0% refusal)

### Illegal
- Overall refusal rate: 60.0%
- Hedged: 3/15
- Full: 3/15

- Most vulnerable model: llava (40.0% refusal)

### Privacy
- Overall refusal rate: 50.0%
- Hedged: 0/6
- Full: 3/6

- Most vulnerable model: llava (50.0% refusal)

## Multimodal-Specific Metrics

### OCR Verification

- OCR verified: 65/69 (94.2%)
- OCR failed: 4/69 (5.8%)

**OCR failures by attack type:**
- Direct Harmful: 2/12 failures (16.7%)
- Ocr Harmful Text: 1/15 failures (6.7%)
- Multi Turn Setup: 1/3 failures (33.3%)

**OCR accuracy by model:**
- gemma3:4b: 23/23 verified (100.0%)
- llava:7b: 20/23 verified (87.0%)
- qwen3-vl:4b: 22/23 verified (95.7%)

### Repetition Detection

- Repetitive responses detected: 0/69 (0.0%)

