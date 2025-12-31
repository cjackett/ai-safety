#!/bin/bash
# Convenience script to run guardrail tests

cd "$(dirname "$0")"

echo "Testing guardrail configurations..."
echo ""

# Quick test mode by default
MODE="${1:-quick}"

if [ "$MODE" = "quick" ]; then
    echo "Running quick tests (--quick mode)..."

    # Test balanced mode
    python test_guardrails.py --config configs/balanced_mode.yaml --quick

elif [ "$MODE" = "full" ]; then
    echo "Running full test suite..."

    # Test all configurations
    for config in configs/*.yaml; do
        echo ""
        echo "=========================================="
        echo "Testing: $config"
        echo "=========================================="
        python test_guardrails.py --config "$config"
    done

elif [ "$MODE" = "analyse" ]; then
    echo "Running analysis on existing results..."
    python analyse_results.py

else
    echo "Usage: $0 [quick|full|analyse]"
    echo "  quick   - Run quick validation test (default)"
    echo "  full    - Run full test suite on all configs"
    echo "  analyse - Analyse existing results"
fi
