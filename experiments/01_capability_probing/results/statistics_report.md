# Capability Probing Analysis Report

Total prompts analysed: 320
Models tested: 4
Harm categories: 8

## Overall Compliance Distribution

- refused: 279 (87.19%)
- hedged: 25 (7.81%)
- full: 16 (5.0%)

## Model Performance


### llama3.2:3b
- Total: 80
- Refused: 80 (100.0%)
- Hedged: 0 (0.0%)
- Full: 0 (0.0%)

### qwen3:4b
- Total: 80
- Refused: 72 (90.0%)
- Hedged: 3 (3.8%)
- Full: 5 (6.2%)

### gemma3:4b
- Total: 80
- Refused: 75 (93.8%)
- Hedged: 3 (3.8%)
- Full: 2 (2.5%)

### mistral:7b
- Total: 80
- Refused: 52 (65.0%)
- Hedged: 19 (23.8%)
- Full: 9 (11.2%)

## Category Analysis


### cbrn
- Refusal rate: 82.5%
- Most vulnerable model: qwen3:4b

### child_safety
- Refusal rate: 92.5%
- Most vulnerable model: qwen3:4b

### cyber
- Refusal rate: 80.0%
- Most vulnerable model: mistral:7b

### deception
- Refusal rate: 90.0%
- Most vulnerable model: mistral:7b

### disinformation
- Refusal rate: 82.5%
- Most vulnerable model: mistral:7b

### harmful_content
- Refusal rate: 95.0%
- Most vulnerable model: qwen3:4b

### illegal
- Refusal rate: 90.0%
- Most vulnerable model: mistral:7b

### privacy
- Refusal rate: 85.0%
- Most vulnerable model: gemma3:4b