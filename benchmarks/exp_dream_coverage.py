#!/usr/bin/env python3
"""
Experiment 3: Proactive Coverage via Dreaming

Question: Do synthetic corruptions anticipate real errors?

This experiment learns concepts, generates dream variants, then tests
whether real corruptions match the pre-registered fixes.
"""

import argparse
import random
from typing import Dict, List, Tuple

from experience_memory import (
    FixRegistry,
    DreamGenerator,
    ErrorSignature,
    ErrorSeverity,
    ErrorType,
)


def generate_concepts(n: int) -> List[Dict]:
    """Generate n synthetic concepts to learn."""
    base_elements = [
        "boundary", "inside", "outside", "center", "edge",
        "before", "after", "during", "start", "end",
        "hot", "cold", "fast", "slow", "big", "small",
        "up", "down", "left", "right", "near", "far",
        "good", "bad", "true", "false", "same", "different",
        "agent", "patient", "cause", "effect", "action",
    ]

    categories = ["spatial", "temporal", "physical", "cognitive", "relational", "abstract"]

    concepts = []
    for i in range(n):
        # Each concept has 3-6 elements with varying activations
        num_elements = random.randint(3, 6)
        elements = random.sample(base_elements, num_elements)
        activations = {el: random.uniform(0.5, 1.0) for el in elements}

        concept = {
            "term": f"concept_{i}",
            "activations": activations,
            "categories": random.sample(categories, random.randint(1, 3)),
        }
        concepts.append(concept)

    return concepts


def apply_real_corruption(
    concept: Dict,
    corruption_type: str
) -> Tuple[Dict, ErrorSignature]:
    """
    Apply a "real" corruption to a concept and return the corrupted
    state plus the error signature that should match.
    """
    activations = concept["activations"].copy()
    categories = concept["categories"]

    if corruption_type == "removal":
        # Remove 1-2 random elements
        if len(activations) > 1:
            to_remove = random.sample(list(activations.keys()),
                                      min(2, len(activations) - 1))
            for key in to_remove:
                del activations[key]
            error_type = ErrorType.REMOVAL
        else:
            error_type = ErrorType.REMOVAL

    elif corruption_type == "intensity":
        # Scale values down
        scale = random.choice([0.3, 0.5, 0.7])
        activations = {k: v * scale for k, v in activations.items()}
        error_type = ErrorType.INTENSITY_SHIFT

    elif corruption_type == "substitution":
        # Replace some elements with random ones
        if activations:
            key = random.choice(list(activations.keys()))
            del activations[key]
            activations["substituted_element"] = random.uniform(0.5, 1.0)
        error_type = ErrorType.SUBSTITUTION

    else:
        error_type = ErrorType.REMOVAL

    # Compute delta (simplified)
    original_sum = sum(concept["activations"].values())
    corrupted_sum = sum(activations.values())
    delta = abs(original_sum - corrupted_sum) / max(original_sum, 0.001)
    delta = min(1.0, delta)

    # Determine severity
    if delta > 0.9:
        severity = ErrorSeverity.CATASTROPHIC
    elif delta > 0.5:
        severity = ErrorSeverity.MAJOR
    elif delta > 0.1:
        severity = ErrorSeverity.MODERATE
    else:
        severity = ErrorSeverity.MINOR

    signature = ErrorSignature(
        severity=severity,
        error_type=error_type,
        context=concept["term"],
        affected_categories=categories,
        delta=delta,
    )

    return activations, signature


def run_coverage_experiment(
    concepts: List[Dict],
    strategies: List[str],
    variants_per_strategy: int,
    corruptions_per_concept: int,
) -> Dict:
    """
    1. Dream variants for each concept
    2. Apply real corruptions
    3. Measure hit rate on pre-registered fixes
    """
    registry = FixRegistry()
    dreamer = DreamGenerator(registry=registry)

    # Phase 1: Dream (pre-register fixes)
    print("\nPhase 1: Dreaming synthetic variants...")
    total_dreamed = 0
    for concept in concepts:
        examples = dreamer.dream(
            concept=concept,
            strategies=strategies,
            variants_per_strategy=variants_per_strategy,
        )
        total_dreamed += len(examples)

    registry_after_dreaming = registry.get_stats()["total_entries"]
    print(f"  Dreamed {total_dreamed} variants")
    print(f"  Registry size: {registry_after_dreaming}")

    # Phase 2: Apply real corruptions and test
    print("\nPhase 2: Testing against real corruptions...")
    hits = 0
    misses = 0
    results_by_type = {
        "removal": {"hits": 0, "misses": 0},
        "intensity": {"hits": 0, "misses": 0},
        "substitution": {"hits": 0, "misses": 0},
    }

    corruption_types = ["removal", "intensity", "substitution"]

    for concept in concepts:
        for _ in range(corruptions_per_concept):
            corruption_type = random.choice(corruption_types)
            corrupted, signature = apply_real_corruption(concept, corruption_type)

            fix = registry.lookup(signature)
            if fix is not None:
                hits += 1
                results_by_type[corruption_type]["hits"] += 1
            else:
                misses += 1
                results_by_type[corruption_type]["misses"] += 1

    return {
        "concepts": len(concepts),
        "strategies": strategies,
        "variants_per_strategy": variants_per_strategy,
        "total_dreamed": total_dreamed,
        "registry_size": registry_after_dreaming,
        "corruptions_tested": hits + misses,
        "hits": hits,
        "misses": misses,
        "hit_rate": hits / (hits + misses) if (hits + misses) > 0 else 0,
        "by_type": results_by_type,
    }


def print_results(results: Dict):
    """Print coverage analysis."""
    print("\n" + "="*70)
    print("RESULTS: Proactive Coverage via Dreaming")
    print("="*70)

    print(f"\nExperiment Configuration:")
    print(f"  Concepts learned:        {results['concepts']}")
    print(f"  Dream strategies:        {results['strategies']}")
    print(f"  Variants per strategy:   {results['variants_per_strategy']}")

    print(f"\nDreaming Phase:")
    print(f"  Total dreamed variants:  {results['total_dreamed']}")
    print(f"  Registry entries:        {results['registry_size']}")

    print(f"\nCoverage Results:")
    print(f"  Corruptions tested:      {results['corruptions_tested']}")
    print(f"  Cache hits:              {results['hits']}")
    print(f"  Cache misses:            {results['misses']}")
    print(f"  Overall hit rate:        {100*results['hit_rate']:.1f}%")

    print(f"\nHit Rate by Corruption Type:")
    for ctype, stats in results["by_type"].items():
        total = stats["hits"] + stats["misses"]
        if total > 0:
            rate = 100 * stats["hits"] / total
            print(f"  {ctype:15} {rate:6.1f}% ({stats['hits']}/{total})")

    print("\n" + "-"*70)
    print("Interpretation:")
    print("  - High hit rate = dreaming anticipates real errors well")
    print("  - Removal/intensity strategies often have higher coverage")
    print("  - Substitution is harder to predict (context-dependent)")
    print("  - >70% coverage indicates effective proactive defense")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Test proactive coverage via dreaming")
    parser.add_argument("--concepts", type=int, default=50, help="Number of concepts")
    parser.add_argument("--strategies", type=str, default="removal,intensity,substitution",
                        help="Comma-separated strategies")
    parser.add_argument("--variants", type=int, default=3, help="Variants per strategy")
    parser.add_argument("--corruptions", type=int, default=5,
                        help="Corruptions to test per concept")
    args = parser.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",")]

    print("="*70)
    print("Experiment 3: Proactive Coverage via Dreaming")
    print("="*70)
    print(f"Concepts:             {args.concepts}")
    print(f"Strategies:           {strategies}")
    print(f"Variants/strategy:    {args.variants}")
    print(f"Corruptions/concept:  {args.corruptions}")

    concepts = generate_concepts(args.concepts)

    results = run_coverage_experiment(
        concepts=concepts,
        strategies=strategies,
        variants_per_strategy=args.variants,
        corruptions_per_concept=args.corruptions,
    )

    print_results(results)


if __name__ == "__main__":
    main()
