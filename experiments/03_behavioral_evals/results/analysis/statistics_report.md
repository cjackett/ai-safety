# Behavioral Safety Evaluation Analysis Report

Total tests: 199
Models tested: 4
Behavioral categories: 5

## Overall Alignment Distribution

- Aligned: 152 (76.4%)
- Concerning: 30 (15.1%)
- Problematic: 17 (8.5%)

## Model Performance


### gemma3:4b
- Total: 50
- Aligned: 48 (96.0%)
- Concerning: 2 (4.0%)
- Problematic: 0 (0.0%)

### llama3.2:3b
- Total: 50
- Aligned: 48 (96.0%)
- Concerning: 1 (2.0%)
- Problematic: 1 (2.0%)

### mistral:7b
- Total: 50
- Aligned: 23 (46.0%)
- Concerning: 16 (32.0%)
- Problematic: 11 (22.0%)

### qwen3:4b
- Total: 49
- Aligned: 33 (67.3%)
- Concerning: 11 (22.4%)
- Problematic: 5 (10.2%)

## Category Analysis


### Deception
- Overall alignment rate: 85.0%
- Concerning: 5/40
- Problematic: 1/40

- Most vulnerable model: mistral:7b (50.0% aligned)

### Honesty
- Overall alignment rate: 61.5%
- Concerning: 9/39
- Problematic: 6/39

- Most vulnerable model: mistral:7b (40.0% aligned)

### Power Seeking
- Overall alignment rate: 77.5%
- Concerning: 4/40
- Problematic: 5/40

- Most vulnerable model: mistral:7b (40.0% aligned)

### Situational Awareness
- Overall alignment rate: 85.0%
- Concerning: 4/40
- Problematic: 2/40

- Most vulnerable model: qwen3:4b (60.0% aligned)

### Sycophancy
- Overall alignment rate: 72.5%
- Concerning: 8/40
- Problematic: 3/40

- Most vulnerable model: mistral:7b (20.0% aligned)
