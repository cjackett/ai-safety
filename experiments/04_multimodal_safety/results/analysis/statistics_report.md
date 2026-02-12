# Multimodal Safety Evaluation Analysis Report

Total tests: 66
Models tested: 3
Attack types: 11
Harm categories: 7

## Overall Compliance Distribution

- refused: 38 (57.6%)
- hedged: 5 (7.6%)
- full: 23 (34.8%)

## Model Performance


### gemma3:4b
- Total: 22
- Refused: 15 (68.2%)
- Hedged: 3 (13.6%)
- Full: 4 (18.2%)

### llava:7b
- Total: 22
- Refused: 12 (54.5%)
- Hedged: 2 (9.1%)
- Full: 8 (36.4%)

### qwen3-vl:4b
- Total: 22
- Refused: 11 (50.0%)
- Hedged: 0 (0.0%)
- Full: 11 (50.0%)

## Attack Type Effectiveness


### Base64 In Image
- Total tests: 3
- Success rate: 66.7% (0 hedged + 2 full)
- Refusal rate: 33.3%

### Caesar Cipher
- Total tests: 3
- Success rate: 66.7% (0 hedged + 2 full)
- Refusal rate: 33.3%

### Cross Modal Inconsistency
- Total tests: 15
- Success rate: 46.7% (1 hedged + 6 full)
- Refusal rate: 53.3%

### Direct Harmful
- Total tests: 12
- Success rate: 33.3% (2 hedged + 2 full)
- Refusal rate: 66.7%

### Educational Framing
- Total tests: 3
- Success rate: 0.0% (0 hedged + 0 full)
- Refusal rate: 100.0%

### Leetspeak In Image
- Total tests: 3
- Success rate: 66.7% (0 hedged + 2 full)
- Refusal rate: 33.3%

### Multi Turn Setup
- Total tests: 3
- Success rate: 100.0% (0 hedged + 3 full)
- Refusal rate: 0.0%

### Ocr Harmful Text
- Total tests: 15
- Success rate: 26.7% (2 hedged + 2 full)
- Refusal rate: 73.3%

### Reverse Text
- Total tests: 3
- Success rate: 66.7% (0 hedged + 2 full)
- Refusal rate: 33.3%

### Roleplay In Image
- Total tests: 3
- Success rate: 33.3% (0 hedged + 1 full)
- Refusal rate: 66.7%

### Rot13 In Image
- Total tests: 3
- Success rate: 33.3% (0 hedged + 1 full)
- Refusal rate: 66.7%

## Category Analysis


### Cbrn
- Overall refusal rate: 50.0%
- Hedged: 1/6
- Full: 2/6

- Most vulnerable model: llava (50.0% refusal)

### Child Safety
- Overall refusal rate: 66.7%
- Hedged: 0/3
- Full: 1/3

- Most vulnerable model: qwen3-vl (0.0% refusal)

### Cyber
- Overall refusal rate: 58.3%
- Hedged: 0/12
- Full: 5/12

- Most vulnerable model: qwen3-vl (50.0% refusal)

### Deception
- Overall refusal rate: 22.2%
- Hedged: 1/9
- Full: 6/9

- Most vulnerable model: qwen3-vl (0.0% refusal)

### Harmful Content
- Overall refusal rate: 73.3%
- Hedged: 2/15
- Full: 2/15

- Most vulnerable model: llava (40.0% refusal)

### Illegal
- Overall refusal rate: 60.0%
- Hedged: 0/15
- Full: 6/15

- Most vulnerable model: llava (40.0% refusal)

### Privacy
- Overall refusal rate: 66.7%
- Hedged: 1/6
- Full: 1/6

- Most vulnerable model: qwen3-vl (50.0% refusal)

## Multimodal-Specific Metrics

### OCR Verification

- OCR verified: 60/69 (87.0%)
- OCR failed: 9/69 (13.0%)

**OCR failures by attack type:**
- Ocr Harmful Text: 3/15 failures (20.0%)
- Direct Harmful: 2/12 failures (16.7%)
- Multi Turn Setup: 2/3 failures (66.7%)
- Roleplay In Image: 1/3 failures (33.3%)
- Educational Framing: 1/3 failures (33.3%)

**OCR accuracy by model:**
- gemma3:4b: 22/23 verified (95.7%)
- llava:7b: 17/23 verified (73.9%)
- qwen3-vl:4b: 21/23 verified (91.3%)

### Repetition Detection

- Repetitive responses detected: 0/69 (0.0%)

