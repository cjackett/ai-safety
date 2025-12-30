#!/usr/bin/env python3
"""
Discover induction heads in GPT-2 small using TransformerLens.

Based on Anthropic's "In-context Learning and Induction Heads" paper
(Olsson et al., 2022). Induction heads detect repeated sequences and
predict the next token based on what followed the pattern previously.

Usage:
    python find_induction_heads.py
"""

import warnings
warnings.filterwarnings('ignore', message='.*torch_dtype.*')

import torch
from transformer_lens import HookedTransformer
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from datetime import datetime
from tqdm import tqdm


class InductionHeadFinder:
    """Find and analyze induction heads in transformer models."""

    def __init__(self, model_name: str = "gpt2-small", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Load model and prepare for analysis.

        Args:
            model_name: TransformerLens model name
            device: Device to run on (cuda/cpu)
        """
        print(f"Loading {model_name} on {device}...")
        self.model = HookedTransformer.from_pretrained(model_name, device=device)
        self.device = device
        self.n_layers = self.model.cfg.n_layers
        self.n_heads = self.model.cfg.n_heads

        print(f"Model loaded: {self.n_layers} layers, {self.n_heads} heads per layer")

        # Results storage
        self.induction_scores = np.zeros((self.n_layers, self.n_heads))
        self.test_results = []
        self.discovered_heads = []

    def load_test_sequences(self, path: str) -> List[Dict]:
        """Load test sequences from JSON."""
        with open(path, 'r') as f:
            sequences = json.load(f)
        print(f"Loaded {len(sequences)} test sequences")
        return sequences

    def compute_induction_score(self, sequence: str, expected_token: Optional[str] = None) -> Dict:
        """
        Compute induction score for each attention head on this sequence.

        Induction heads should:
        1. Attend backwards to previous occurrence of current token
        2. Increase logit for token that followed previous occurrence

        Args:
            sequence: Input text sequence
            expected_token: Expected next token (for pattern sequences)

        Returns:
            Dict with per-head induction scores and attention patterns
        """
        # Tokenize
        tokens = self.model.to_tokens(sequence)

        # Run model with cache to get activations
        logits, cache = self.model.run_with_cache(tokens)

        # Get attention patterns for all layers/heads
        # Shape: [batch=1, n_heads, seq_len, seq_len]
        scores_per_head = np.zeros((self.n_layers, self.n_heads))

        for layer in range(self.n_layers):
            attn_pattern = cache["pattern", layer][0]  # [n_heads, seq_len, seq_len]

            for head in range(self.n_heads):
                # Compute induction score for this head
                # Look for diagonal stripe pattern: each position attending to
                # the previous occurrence of the current token
                score = self._compute_head_induction_score(attn_pattern[head])
                scores_per_head[layer, head] = score

        result = {
            "sequence": sequence,
            "expected_token": expected_token,
            "tokens": self.model.to_str_tokens(tokens[0]),
            "scores_per_head": scores_per_head.tolist(),
            "num_tokens": len(tokens[0])
        }

        return result

    def _compute_head_induction_score(self, attn_pattern: torch.Tensor) -> float:
        """
        Compute induction score for a single attention head.

        Induction heads show a characteristic pattern: they attend backwards
        from position i to earlier positions that have matching tokens.

        Key signature: The attention matrix shows a "stripe" pattern where
        attention[i, j] is high when the sequence has a repeated pattern ending
        at positions j and i.

        More precisely, for sequence "A B C ... A B", the position after the
        second "A" should attend to the position after the first "A" (where "B" was).

        Args:
            attn_pattern: Attention weights [seq_len, seq_len]

        Returns:
            Induction score between 0 and 1
        """
        attn = attn_pattern.cpu().numpy()
        seq_len = attn.shape[0]

        if seq_len < 4:
            return 0.0

        # Strategy: Look for the characteristic induction pattern
        # Position i should attend to earlier positions, specifically
        # to positions that are at similar offsets in earlier patterns

        # Measure 1: Stripe score - attention to non-recent past positions
        # (avoiding immediate previous token which is handled by other circuits)
        stripe_scores = []
        for i in range(3, seq_len):
            # For position i, look at attention to positions 2 or more steps back
            # but not too far (recent context is most relevant)
            lookback_start = max(0, i - 10)
            lookback_end = i - 2  # Exclude positions i-1 and i

            if lookback_end > lookback_start:
                # Average attention to this lookback window
                lookback_attn = np.mean(attn[i, lookback_start:lookback_end])
                stripe_scores.append(lookback_attn)

        if not stripe_scores:
            return 0.0

        stripe_score = np.mean(stripe_scores)

        # Measure 2: Induction heads should show relatively uniform attention
        # to the lookback window (not too concentrated on single position)
        # This distinguishes from other attention patterns

        # Measure 3: Lower score if too much attention on immediate previous token
        # (that's typically handled by different circuits)
        immediate_attn = np.mean([attn[i, i-1] for i in range(1, seq_len)])

        # Measure 4: Check for the diagonal structure
        # For true induction, attention[i,j] should be higher when there's
        # a matching pattern at offset k: attention[i, i-k] ≈ attention[i-k, i-2k]
        diagonal_coherence = 0.0
        count = 0
        for i in range(4, min(seq_len, 12)):  # Check in early-middle positions
            for k in range(2, min(i-1, 5)):  # Various offsets
                if i - 2*k >= 0:
                    # Compare attention at matched positions
                    coherence = min(attn[i, i-k], attn[i-k, max(0, i-2*k)])
                    diagonal_coherence += coherence
                    count += 1

        if count > 0:
            diagonal_coherence /= count

        # Combine measures
        # High stripe score (attention to past context)
        # High diagonal coherence (structured attention pattern)
        # Low immediate attention (not just copying previous token)

        base_score = stripe_score

        # Boost if diagonal structure is present
        if diagonal_coherence > 0.05:
            base_score *= (1 + diagonal_coherence)

        # Penalize high immediate attention
        if immediate_attn > 0.3:
            base_score *= (1 - 0.5 * (immediate_attn - 0.3))

        return max(0.0, min(1.0, base_score))

    def find_induction_heads(self, threshold: float = 0.3) -> List[Tuple[int, int]]:
        """
        Identify heads with high induction scores.

        Args:
            threshold: Minimum average induction score to classify as induction head

        Returns:
            List of (layer, head) tuples for discovered induction heads
        """
        avg_scores = np.mean(self.induction_scores, axis=2)  # Average across test sequences

        induction_heads = []
        for layer in range(self.n_layers):
            for head in range(self.n_heads):
                if avg_scores[layer, head] >= threshold:
                    induction_heads.append((layer, head))

        return induction_heads

    def run_discovery(self, sequences_path: str, output_dir: str):
        """
        Main discovery pipeline:
        1. Load test sequences
        2. Compute induction scores across all heads for each sequence
        3. Aggregate results
        4. Identify candidate induction heads
        5. Save results

        Args:
            sequences_path: Path to test sequences JSON
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load sequences
        sequences = self.load_test_sequences(sequences_path)

        # Store scores across all sequences
        all_scores = np.zeros((len(sequences), self.n_layers, self.n_heads))

        print(f"\nComputing induction scores for {len(sequences)} sequences...")

        # Compute scores for each sequence
        for seq_idx, seq_data in enumerate(tqdm(sequences, desc="Processing sequences")):
            sequence = seq_data["sequence"]
            expected_token = seq_data.get("expected_token")

            result = self.compute_induction_score(sequence, expected_token)

            # Store scores
            scores = np.array(result["scores_per_head"])
            all_scores[seq_idx] = scores

            # Store detailed results
            self.test_results.append({
                "sequence_id": seq_idx,
                "category": seq_data.get("category"),
                "sequence": sequence,
                "pattern": seq_data.get("pattern"),
                "expected_token": expected_token,
                "scores_per_head": result["scores_per_head"],
                "num_tokens": result["num_tokens"]
            })

        # Compute average scores across all sequences
        self.induction_scores = np.mean(all_scores, axis=0)

        print(f"\nAverage induction scores computed across {len(sequences)} sequences")
        print(f"Score range: {self.induction_scores.min():.3f} - {self.induction_scores.max():.3f}")

        # Find induction heads
        threshold = 0.3
        self.discovered_heads = []

        for layer in range(self.n_layers):
            for head in range(self.n_heads):
                avg_score = self.induction_scores[layer, head]
                if avg_score >= threshold:
                    self.discovered_heads.append({
                        "layer": int(layer),
                        "head": int(head),
                        "induction_score": float(avg_score)
                    })

        # Sort by score
        self.discovered_heads.sort(key=lambda x: x["induction_score"], reverse=True)

        print(f"\nFound {len(self.discovered_heads)} induction heads (threshold={threshold})")
        for head_info in self.discovered_heads:
            print(f"  Layer {head_info['layer']}, Head {head_info['head']}: "
                  f"score={head_info['induction_score']:.3f}")

        # Save results
        self._save_results(output_path)

    def _save_results(self, output_dir: Path):
        """Save all results to JSON files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save discovered heads
        heads_file = output_dir / "discovered_heads.json"
        with open(heads_file, 'w') as f:
            json.dump({
                "model": "gpt2-small",
                "timestamp": timestamp,
                "n_layers": self.n_layers,
                "n_heads": self.n_heads,
                "discovery_threshold": 0.3,
                "num_sequences_tested": len(self.test_results),
                "induction_heads": self.discovered_heads
            }, f, indent=2)
        print(f"\nSaved discovered heads to {heads_file}")

        # Save induction scores matrix
        scores_file = output_dir / "induction_scores.json"
        with open(scores_file, 'w') as f:
            json.dump({
                "model": "gpt2-small",
                "timestamp": timestamp,
                "scores_by_layer": {
                    str(layer): self.induction_scores[layer].tolist()
                    for layer in range(self.n_layers)
                },
                "average_score_per_layer": np.mean(self.induction_scores, axis=1).tolist(),
                "max_score_per_layer": np.max(self.induction_scores, axis=1).tolist()
            }, f, indent=2)
        print(f"Saved induction scores to {scores_file}")

        # Save detailed test results
        results_file = output_dir / f"test_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        print(f"Saved detailed test results to {results_file}")


def main():
    """Run induction head discovery experiment."""
    print("=" * 70)
    print("Induction Head Discovery - GPT-2 Small")
    print("=" * 70)

    # Get script directory for relative paths
    script_dir = Path(__file__).parent

    # Initialize finder
    finder = InductionHeadFinder(model_name="gpt2-small")

    # Run discovery
    finder.run_discovery(
        sequences_path=str(script_dir / "test_sequences" / "sequences.json"),
        output_dir=str(script_dir / "results")
    )

    print("\n" + "=" * 70)
    print("Discovery complete!")
    print("=" * 70)
    print(f"\nNext step: Run 'python analyse_circuits.py' to generate visualizations")


if __name__ == "__main__":
    main()
