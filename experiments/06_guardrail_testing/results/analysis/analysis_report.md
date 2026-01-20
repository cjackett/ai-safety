# Guardrail Testing Analysis Report
Generated: /home/chris/Dev/ai-safety/experiments/06_guardrail_testing/results/analysis
---

## Configuration: Strict

### Jailbreak Resistance

- **Total Tests**: 44
- **Blocked at Input**: 9 (20.5%)
- **Blocked at Output**: 16 (36.4%)
- **Bypassed**: 19 (43.2%)

### False Positives

- **Total Tests**: 50
- **False Positives**: 5 (10.0%)
- **Correctly Allowed**: 45

### Latency Performance

- **Mean**: 49469.4 ms
- **Median**: 48336.6 ms
- **P95**: 59676.1 ms
- **P99**: 59736.6 ms

---

## Configuration: Permissive

### Jailbreak Resistance

- **Total Tests**: 44
- **Blocked at Input**: 0 (0.0%)
- **Blocked at Output**: 9 (20.5%)
- **Bypassed**: 35 (79.5%)

### False Positives

- **Total Tests**: 50
- **False Positives**: 0 (0.0%)
- **Correctly Allowed**: 50

### Latency Performance

- **Mean**: 39291.2 ms
- **Median**: 38832.9 ms
- **P95**: 46887.8 ms
- **P99**: 47840.7 ms

---

## Configuration: Balanced

### Jailbreak Resistance

- **Total Tests**: 44
- **Blocked at Input**: 6 (13.6%)
- **Blocked at Output**: 16 (36.4%)
- **Bypassed**: 22 (50.0%)

### False Positives

- **Total Tests**: 50
- **False Positives**: 4 (8.0%)
- **Correctly Allowed**: 46

### Latency Performance

- **Mean**: 46946.6 ms
- **Median**: 46982.1 ms
- **P95**: 55500.7 ms
- **P99**: 59116.5 ms

---

