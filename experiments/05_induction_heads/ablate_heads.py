#!/usr/bin/env python3
"""
Ablation studies to causally verify induction heads' contribution to in-context learning.

Tests the hypothesis that induction heads are causally responsible for in-context
learning by "knocking out" specific attention heads and measuring the impact on
the model's ability to predict tokens later in the context vs. earlier.

Based on Olsson et al. (2022) Argument 3: Direct ablation shows that removing
induction heads greatly decreases in-context learning in small models.

Usage:
    python ablate_heads.py
"""

import warnings
warnings.filterwarnings('ignore', message='.*torch_dtype.*')

import torch
from transformer_lens import HookedTransformer
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
from functools import partial


class HeadAblator:
    """Ablate attention heads and measure impact on in-context learning."""

    def __init__(self, model_name: str = "gpt2-small", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Load model and prepare for ablation studies.

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
        self.ablation_results = []

    def compute_icl_score(self, texts: List[str], ablate_layer: int = None, ablate_head: int = None) -> float:
        """
        Compute in-context learning score: loss[token_50] - loss[token_500].

        Following Olsson et al. (2022), this measures how much better the model
        gets at prediction as it sees more context.

        Args:
            texts: List of text sequences to evaluate on
            ablate_layer: If specified, ablate this layer's head
            ablate_head: If specified, ablate this head (requires ablate_layer)

        Returns:
            In-context learning score (higher = more in-context learning)
        """
        scores = []

        for text in texts:
            tokens = self.model.to_tokens(text)

            # Skip if sequence too short
            if tokens.shape[1] < 500:
                continue

            # Set up ablation hook if needed
            if ablate_layer is not None and ablate_head is not None:
                # Create hook to zero out specific head's contribution
                # We zero out the head's output after the attention mechanism
                def ablation_hook(activations, hook):
                    # activations shape: [batch, pos, n_heads, d_head]
                    # Zero out the specified head
                    activations[:, :, ablate_head, :] = 0.0
                    return activations

                # Run with ablation - use hook_z which is the attention output before W_O
                hook_name = f"blocks.{ablate_layer}.attn.hook_z"
                with self.model.hooks([(hook_name, ablation_hook)]):
                    logits = self.model(tokens)
            else:
                # Run normally
                logits = self.model(tokens)

            # Compute loss at different positions
            # Position 50 (early context)
            loss_50 = self._compute_loss_at_position(logits, tokens, 50)

            # Position 500 (late context)
            loss_500 = self._compute_loss_at_position(logits, tokens, 500)

            # In-context learning score: how much loss decreases from early to late
            icl_score = loss_50 - loss_500
            scores.append(icl_score)

        return np.mean(scores)

    def _compute_loss_at_position(self, logits: torch.Tensor, tokens: torch.Tensor, position: int) -> float:
        """
        Compute cross-entropy loss for prediction at a specific position.

        Args:
            logits: Model output [batch, seq_len, vocab_size]
            tokens: Input tokens [batch, seq_len]
            position: Token position to compute loss for

        Returns:
            Loss value at that position
        """
        # Get logits for predicting token at 'position'
        # (using context up to position-1)
        pred_logits = logits[0, position - 1, :]

        # Get actual token at position
        target_token = tokens[0, position]

        # Compute cross-entropy loss
        loss = torch.nn.functional.cross_entropy(
            pred_logits.unsqueeze(0),
            target_token.unsqueeze(0)
        )

        return loss.item()

    def load_test_texts(self, sequences_path: str = None) -> List[str]:
        """
        Create test texts for in-context learning measurement.

        We need texts with 500+ tokens to measure ICL at position 50 vs 500.
        We'll create synthetic long texts with repeated patterns to test induction.
        """
        texts = []

        # Create synthetic repeated texts to test induction behavior
        # Pattern 1: Repeated story with names
        story1 = """
        Once upon a time, there was a young wizard named Harry Potter who lived with his aunt and uncle.
        Harry Potter discovered he had magical powers when strange things started happening around him.
        His friend Hermione Granger was the brightest witch of her age and helped Harry Potter often.
        Ron Weasley, another friend, came from a large wizarding family and stood by Harry Potter always.
        The headmaster Albus Dumbledore guided Harry Potter through many challenges at the school.
        Professor Snape seemed to dislike Harry Potter but had complex reasons for his behavior.
        Lord Voldemort, the dark wizard, sought to destroy Harry Potter and take over the wizarding world.
        Harry Potter faced many trials but grew stronger with each challenge he overcame.
        """

        # Pattern 2: Technical documentation with repetition
        story2 = """
        The Python programming language is widely used for data science applications.
        Python programming language supports multiple paradigms including object-oriented and functional.
        Machine learning frameworks like TensorFlow are built using the Python programming language.
        Scientists prefer Python programming language because of its simple syntax and powerful libraries.
        Web development with Django demonstrates how Python programming language can scale.
        The community around Python programming language continues to grow each year.
        Automation scripts written in Python programming language save countless hours of work.
        Python programming language will remain popular for years to come.
        """

        # Pattern 3: Repeated sequences with clear patterns
        story3 = """
        The cat sat on the mat. The dog ran in the park. The bird flew in the sky.
        The cat sat on the mat again. The dog ran in the park again. The bird flew in the sky again.
        Every morning, the cat sat on the mat while watching the world. Every morning, the dog ran in the park.
        Children love when the cat sat on the mat and played with them. Children love when the dog ran in the park.
        At sunset, the cat sat on the mat one last time. At sunset, the dog ran in the park one last time.
        The bird flew in the sky throughout the day, watching everything below.
        """

        # Pattern 4: News-style repeated references
        story4 = """
        The President announced a new policy today. The President said the policy would help millions.
        Congress debated the President's proposal extensively. Many senators support the President's initiative.
        The President traveled to five states to promote the new policy. Citizens asked the President questions.
        Media outlets covered the President's tour across the nation. The President answered criticism.
        International leaders called the President to discuss the implications. The President reassured allies.
        Next week, the President will address the nation about the policy's progress.
        """

        # Repeat each story multiple times to create long sequences
        for story in [story1, story2, story3, story4]:
            # Repeat 5 times to create ~500-1000 token sequences
            long_text = (story + " ") * 5
            texts.append(long_text)

        # Also create some mixed patterns
        mixed = (story1 + " " + story2 + " " + story3 + " ") * 2
        texts.append(mixed)

        print(f"Created {len(texts)} test texts for ICL measurement")

        # Verify length
        for i, text in enumerate(texts):
            tokens = self.model.to_tokens(text)
            print(f"  Text {i+1}: {tokens.shape[1]} tokens")

        return texts

    def ablate_head(self, layer: int, head: int, test_texts: List[str]) -> Dict:
        """
        Ablate a specific head and measure impact on in-context learning.

        Args:
            layer: Layer index
            head: Head index within layer
            test_texts: List of texts to evaluate on

        Returns:
            Dictionary with ablation results
        """
        # Compute baseline ICL score (no ablation)
        baseline_score = self.compute_icl_score(test_texts)

        # Compute ICL score with this head ablated
        ablated_score = self.compute_icl_score(test_texts, ablate_layer=layer, ablate_head=head)

        # Impact = how much ICL decreased when we ablated this head
        # Positive impact means the head was contributing to ICL
        impact = baseline_score - ablated_score

        result = {
            "layer": layer,
            "head": head,
            "baseline_icl_score": float(baseline_score),
            "ablated_icl_score": float(ablated_score),
            "icl_impact": float(impact),
            "percent_decrease": float((impact / baseline_score * 100) if baseline_score != 0 else 0)
        }

        return result

    def run_ablation_study(
        self,
        sequences_path: str,
        discovered_heads_path: str,
        output_dir: str,
        top_n: int = 10,
        bottom_n: int = 10,
        random_n: int = 10
    ):
        """
        Run ablation study on selected heads.

        Ablates:
        - Top N induction heads (highest scores)
        - Bottom N heads (lowest scores)
        - Random N middle heads (for comparison)

        Args:
            sequences_path: Path to test sequences
            discovered_heads_path: Path to discovered induction heads JSON
            output_dir: Directory to save ablation results
            top_n: Number of top induction heads to ablate
            bottom_n: Number of bottom heads to ablate
            random_n: Number of random middle heads to ablate
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load test texts (sequences_path not used - we generate synthetic texts)
        test_texts = self.load_test_texts()

        # Load discovered induction heads
        with open(discovered_heads_path) as f:
            heads_data = json.load(f)
            discovered_heads = heads_data["induction_heads"]

        # Load induction scores for all heads
        scores_path = Path(discovered_heads_path).parent / "induction_scores.json"
        with open(scores_path) as f:
            scores_data = json.load(f)

        # Reconstruct all head scores
        all_head_scores = []
        for layer in range(self.n_layers):
            layer_scores = scores_data["scores_by_layer"][str(layer)]
            for head in range(self.n_heads):
                all_head_scores.append({
                    "layer": layer,
                    "head": head,
                    "induction_score": layer_scores[head]
                })

        # Sort by induction score
        all_head_scores.sort(key=lambda x: x["induction_score"], reverse=True)

        # Select heads to ablate
        top_heads = all_head_scores[:top_n]
        bottom_heads = all_head_scores[-bottom_n:]

        # Random middle heads (avoid top and bottom)
        middle_range = all_head_scores[top_n:-bottom_n]
        np.random.seed(42)
        random_indices = np.random.choice(len(middle_range), min(random_n, len(middle_range)), replace=False)
        random_heads = [middle_range[i] for i in random_indices]

        heads_to_ablate = {
            "top_induction": top_heads,
            "random_middle": random_heads,
            "bottom": bottom_heads
        }

        print(f"\n{'='*70}")
        print("Running Ablation Study")
        print(f"{'='*70}\n")
        print(f"Test texts: {len(test_texts)}")
        print(f"Ablating {top_n} top heads, {random_n} random heads, {bottom_n} bottom heads")
        print(f"Total ablations: {top_n + random_n + bottom_n}\n")

        # Compute baseline (no ablation)
        print("Computing baseline in-context learning score...")
        baseline_score = self.compute_icl_score(test_texts)
        print(f"Baseline ICL score: {baseline_score:.4f}\n")

        # Run ablations
        all_results = []

        for category, heads in heads_to_ablate.items():
            print(f"\nAblating {category} heads...")

            for head_info in tqdm(heads, desc=f"Ablating {category}"):
                layer = head_info["layer"]
                head = head_info["head"]

                result = self.ablate_head(layer, head, test_texts)
                result["category"] = category
                result["induction_score"] = head_info["induction_score"]

                all_results.append(result)

        # Sort results by impact
        all_results.sort(key=lambda x: x["icl_impact"], reverse=True)

        # Save results
        results_file = output_path / "ablation_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "baseline_icl_score": float(baseline_score),
                "num_test_texts": len(test_texts),
                "ablation_results": all_results
            }, f, indent=2)

        print(f"\n{'='*70}")
        print("Ablation Study Complete")
        print(f"{'='*70}\n")

        # Print summary
        print("Top 5 most important heads for in-context learning:")
        for i, result in enumerate(all_results[:5], 1):
            print(f"{i}. Layer {result['layer']}, Head {result['head']}: "
                  f"Impact={result['icl_impact']:.4f} ({result['percent_decrease']:.1f}% decrease), "
                  f"Category={result['category']}, "
                  f"Induction score={result['induction_score']:.3f}")

        print(f"\nResults saved to {results_file}")

        self.ablation_results = all_results
        return all_results


def main():
    """Run ablation study on discovered induction heads."""
    print("="*70)
    print("Induction Head Ablation Study - GPT-2 Small")
    print("="*70)

    # Get script directory
    script_dir = Path(__file__).parent

    # Initialize ablator
    ablator = HeadAblator(model_name="gpt2-small")

    # Run ablation study
    ablator.run_ablation_study(
        sequences_path=str(script_dir / "test_sequences" / "sequences.json"),
        discovered_heads_path=str(script_dir / "results" / "discovered_heads.json"),
        output_dir=str(script_dir / "results"),
        top_n=10,      # Top 10 induction heads
        bottom_n=10,   # Bottom 10 heads
        random_n=10    # 10 random middle heads
    )

    print("\n" + "="*70)
    print("Next step: Run 'python analyse_circuits.py' to visualize ablation results")
    print("="*70)


if __name__ == "__main__":
    main()
