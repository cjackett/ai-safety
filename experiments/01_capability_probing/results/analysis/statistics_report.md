# Capability Probing Analysis Report

Total prompts analysed: 320
Models tested: 4
Harm categories: 8

## Overall Compliance Distribution

- refused: 255 (79.69%)
- hedged: 56 (17.5%)
- full: 9 (2.81%)

## Model Performance


### llama3.2:3b
- Total: 80
- Refused: 80 (100.0%)
- Hedged: 0 (0.0%)
- Full: 0 (0.0%)

### qwen3:4b
- Total: 80
- Refused: 69 (86.2%)
- Hedged: 6 (7.5%)
- Full: 5 (6.2%)

### gemma3:4b
- Total: 80
- Refused: 74 (92.5%)
- Hedged: 6 (7.5%)
- Full: 0 (0.0%)

### mistral:7b
- Total: 80
- Refused: 32 (40.0%)
- Hedged: 44 (55.0%)
- Full: 4 (5.0%)

## Category Analysis


### cbrn
- Refusal rate: 82.5%

### child_safety
- Refusal rate: 90.0%
- Most vulnerable model: qwen3:4b

### cyber
- Refusal rate: 70.0%
- Most vulnerable model: mistral:7b

### deception
- Refusal rate: 80.0%
- Most vulnerable model: mistral:7b

### disinformation
- Refusal rate: 72.5%
- Most vulnerable model: mistral:7b

### harmful_content
- Refusal rate: 90.0%

### illegal
- Refusal rate: 77.5%
- Most vulnerable model: qwen3:4b

### privacy
- Refusal rate: 75.0%
- Most vulnerable model: mistral:7b