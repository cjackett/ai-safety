# Guardrail Testing Analysis Report
Generated: /home/chris/Dev/ai-safety/experiments/06_guardrail_testing/results
---

## Configuration: Strict

### Jailbreak Resistance

- **Total Tests**: 44
- **Blocked at Input**: 9 (20.5%)
- **Blocked at Output**: 15 (34.1%)
- **Bypassed**: 20 (45.5%)

### False Positives

- **Total Tests**: 50
- **False Positives**: 3 (6.0%)
- **Correctly Allowed**: 47

### Latency Performance

- **Mean**: 48190.0 ms
- **Median**: 48017.1 ms
- **P95**: 55989.6 ms
- **P99**: 60571.7 ms

---

## Configuration: Permissive

### Jailbreak Resistance

- **Total Tests**: 44
- **Blocked at Input**: 0 (0.0%)
- **Blocked at Output**: 8 (18.2%)
- **Bypassed**: 36 (81.8%)

### False Positives

- **Total Tests**: 50
- **False Positives**: 0 (0.0%)
- **Correctly Allowed**: 50

### Latency Performance

- **Mean**: 39927.4 ms
- **Median**: 39973.1 ms
- **P95**: 47823.2 ms
- **P99**: 51026.8 ms

---

## Configuration: Balanced

### Jailbreak Resistance

- **Total Tests**: 44
- **Blocked at Input**: 6 (13.6%)
- **Blocked at Output**: 10 (22.7%)
- **Bypassed**: 28 (63.6%)

### False Positives

- **Total Tests**: 50
- **False Positives**: 4 (8.0%)
- **Correctly Allowed**: 46

### Latency Performance

- **Mean**: 46468.3 ms
- **Median**: 45470.6 ms
- **P95**: 56251.6 ms
- **P99**: 59295.7 ms

---

