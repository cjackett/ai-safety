# Behavioral Safety Evaluation Analysis Report

Total tests: 200
Models tested: 4
Behavioral categories: 5

## Overall Alignment Distribution

- aligned: 150 (75.0%)
- concerning: 33 (16.5%)
- problematic: 17 (8.5%)

## Model Performance


### gemma3:4b
- Total: 50
- Aligned: 48 (96.0%)
- Concerning: 1 (2.0%)
- Problematic: 1 (2.0%)

### llama3.2:3b
- Total: 50
- Aligned: 49 (98.0%)
- Concerning: 0 (0.0%)
- Problematic: 1 (2.0%)

### mistral:7b
- Total: 50
- Aligned: 22 (44.0%)
- Concerning: 22 (44.0%)
- Problematic: 6 (12.0%)

### qwen3:4b
- Total: 50
- Aligned: 31 (62.0%)
- Concerning: 10 (20.0%)
- Problematic: 9 (18.0%)

## Category Analysis


### Deception
- Overall alignment rate: 82.5%
- Concerning: 7/40
- Problematic: 0/40

- Most vulnerable model: mistral:7b (50.0% aligned)

### Honesty
- Overall alignment rate: 75.0%
- Concerning: 3/40
- Problematic: 7/40

- Most vulnerable model: qwen3:4b (50.0% aligned)

### Power Seeking
- Overall alignment rate: 70.0%
- Concerning: 7/40
- Problematic: 5/40

- Most vulnerable model: mistral:7b (10.0% aligned)

### Situational Awareness
- Overall alignment rate: 80.0%
- Concerning: 6/40
- Problematic: 2/40

- Most vulnerable model: qwen3:4b (50.0% aligned)

### Sycophancy
- Overall alignment rate: 67.5%
- Concerning: 10/40
- Problematic: 3/40

- Most vulnerable model: mistral:7b (10.0% aligned)
