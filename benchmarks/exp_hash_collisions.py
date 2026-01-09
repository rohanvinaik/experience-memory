#!/usr/bin/env python3
"""
Experiment 6: Error Taxonomy Discriminability

Question: Do hash collisions cause fix confusion?

This experiment generates a large number of distinct errors and measures
the hash collision rate to ensure fixes are applied correctly.
"""

import argparse
import random
from collections import defaultdict

from experience_memory import (
    ErrorSignature,
    ErrorSeverity,
    ErrorType,
)


def generate_diverse_signatures(n: int) -> list[ErrorSignature]:
    """Generate n diverse error signatures."""
    signatures = []

    severities = list(ErrorSeverity)
    error_types = list(ErrorType)

    # Generate diverse contexts
    contexts = [f"domain_{i}" for i in range(100)]
    contexts += [f"task_{i}" for i in range(100)]
    contexts += [f"module_{i}" for i in range(100)]

    # Generate diverse category combinations
    base_categories = ["spatial", "temporal", "physical", "cognitive",
                       "social", "emotional", "abstract", "relational"]
    category_combos = []
    for cat in base_categories:
        category_combos.append([cat])
    for i, c1 in enumerate(base_categories):
        for c2 in base_categories[i+1:]:
            category_combos.append([c1, c2])

    for i in range(n):
        sig = ErrorSignature(
            severity=severities[i % len(severities)],
            error_type=error_types[i % len(error_types)],
            context=contexts[i % len(contexts)],
            affected_categories=category_combos[i % len(category_combos)],
            delta=random.uniform(0.0, 1.0),
        )
        signatures.append(sig)

    return signatures


def analyze_collisions(signatures: list[ErrorSignature]) -> dict:
    """Analyze hash collisions among signatures."""
    hash_to_sigs = defaultdict(list)

    for sig in signatures:
        h = sig.hash()
        hash_to_sigs[h].append(sig)

    # Find collisions
    collisions = {h: sigs for h, sigs in hash_to_sigs.items() if len(sigs) > 1}

    # Analyze collision types
    intended_collisions = 0  # Same type/context, different delta
    unintended_collisions = 0  # Different type/context, same hash

    for h, sigs in collisions.items():
        # Check if all signatures in collision group have same structural properties
        first = sigs[0]
        all_same_structure = all(
            s.severity == first.severity and
            s.error_type == first.error_type and
            s.context == first.context and
            sorted(s.affected_categories) == sorted(first.affected_categories)
            for s in sigs
        )

        if all_same_structure:
            intended_collisions += len(sigs) - 1
        else:
            unintended_collisions += len(sigs) - 1

    return {
        "total_signatures": len(signatures),
        "unique_hashes": len(hash_to_sigs),
        "collision_groups": len(collisions),
        "total_colliding_sigs": sum(len(sigs) for sigs in collisions.values()),
        "intended_collisions": intended_collisions,
        "unintended_collisions": unintended_collisions,
        "collision_rate": len(collisions) / len(signatures) if signatures else 0,
        "collision_examples": list(collisions.items())[:3],  # First 3 examples
    }


def print_results(analysis: dict):
    """Print collision analysis results."""
    print("\n" + "="*70)
    print("RESULTS: Error Taxonomy Discriminability")
    print("="*70)

    print(f"\nHash Space Analysis:")
    print(f"  Total signatures generated:  {analysis['total_signatures']:,}")
    print(f"  Unique hashes:               {analysis['unique_hashes']:,}")
    print(f"  Hash utilization:            {100*analysis['unique_hashes']/analysis['total_signatures']:.2f}%")

    print(f"\nCollision Analysis:")
    print(f"  Collision groups:            {analysis['collision_groups']}")
    print(f"  Signatures in collisions:    {analysis['total_colliding_sigs']}")
    print(f"  Intended (same structure):   {analysis['intended_collisions']}")
    print(f"  Unintended (different):      {analysis['unintended_collisions']}")
    print(f"  Unintended collision rate:   {100*analysis['unintended_collisions']/analysis['total_signatures']:.4f}%")

    if analysis['collision_examples']:
        print(f"\nCollision Examples (first {len(analysis['collision_examples'])}):")
        for i, (h, sigs) in enumerate(analysis['collision_examples'][:3]):
            print(f"\n  Group {i+1} (hash: {h}):")
            for j, sig in enumerate(sigs[:3]):
                print(f"    {j+1}. {sig.error_type.value}/{sig.context} "
                      f"[{','.join(sig.affected_categories)}] "
                      f"severity={sig.severity.value}")

    print("\n" + "-"*70)
    print("Interpretation:")
    print("  - Intended collisions: Same error type, different delta (by design)")
    print("  - Unintended collisions: Different errors mapping to same hash (bad)")
    print("  - With 16-char SHA256 (64 bits), expect ~1 collision per 2^32 entries")
    print("  - Unintended rate < 0.01% is acceptable for most applications")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze hash collision rate")
    parser.add_argument(
        "--num-errors",
        type=int,
        default=100000,
        help="Number of errors to generate"
    )
    args = parser.parse_args()

    print("="*70)
    print("Experiment 6: Error Taxonomy Discriminability")
    print("="*70)
    print(f"Generating {args.num_errors:,} error signatures...")

    signatures = generate_diverse_signatures(args.num_errors)

    print("Analyzing hash collisions...")
    analysis = analyze_collisions(signatures)

    print_results(analysis)


if __name__ == "__main__":
    main()
