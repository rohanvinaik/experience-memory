#!/usr/bin/env python3
"""
Experiment 2: Repeated Error Elimination

Question: How quickly do fix lookups eliminate recurring errors?

This experiment simulates a stream of errors where some patterns repeat,
measuring how effectively the registry caches and applies fixes.
"""

import argparse
import random
from collections import defaultdict

from experience_memory import (
    FixRegistry,
    ErrorSignature,
    Fix,
    ErrorSeverity,
    ErrorType,
    FixType,
)


def generate_error_stream(
    length: int,
    num_unique_errors: int,
    repeat_probability: float = 0.7
) -> list[ErrorSignature]:
    """
    Generate a stream of errors with controlled repetition.

    Args:
        length: Total number of errors in stream
        num_unique_errors: Number of distinct error patterns
        repeat_probability: Probability of repeating a previously seen error
    """
    # Create pool of unique errors
    error_pool = []
    for i in range(num_unique_errors):
        sig = ErrorSignature(
            severity=list(ErrorSeverity)[i % 4],
            error_type=list(ErrorType)[i % 7],
            context=f"context_{i % 10}",
            affected_categories=[["spatial", "physical"], ["temporal"], ["cognitive"]][i % 3],
            delta=random.uniform(0.1, 0.9),
        )
        error_pool.append(sig)

    # Generate stream with repeats
    stream = []
    seen_errors = []

    for _ in range(length):
        if seen_errors and random.random() < repeat_probability:
            # Repeat a previously seen error
            error = random.choice(seen_errors)
            # Vary the delta slightly (shouldn't affect hash)
            error = ErrorSignature(
                severity=error.severity,
                error_type=error.error_type,
                context=error.context,
                affected_categories=error.affected_categories,
                delta=random.uniform(0.1, 0.9),  # Different delta
            )
        else:
            # New error from pool
            error = random.choice(error_pool)
            if error not in seen_errors:
                seen_errors.append(error)

        stream.append(error)

    return stream


def generate_fix_for_error(error: ErrorSignature) -> Fix:
    """Generate a deterministic fix based on error type."""
    fix_mapping = {
        ErrorType.REMOVAL: FixType.ADD_ELEMENTS,
        ErrorType.SUBSTITUTION: FixType.SWAP_ELEMENTS,
        ErrorType.INTENSITY_SHIFT: FixType.ADJUST_VALUES,
        ErrorType.STRUCTURAL_SWAP: FixType.RESTORE_CATEGORY,
        ErrorType.RELATIONSHIP_INVERSION: FixType.INVERT_RELATIONSHIP,
        ErrorType.PARTIAL_DEFINITION: FixType.COMPLETE_DEFINITION,
        ErrorType.COMPONENT_REMOVAL: FixType.DECOMPOSE,
    }

    return Fix(
        fix_type=fix_mapping.get(error.error_type, FixType.ADD_ELEMENTS),
        elements_to_add={"restored": 1.0} if error.error_type == ErrorType.REMOVAL else {},
        elements_to_remove=[],
        value_adjustments={},
        category_moves={},
        definition_supplement=None,
        decomposition=None,
    )


def simulate_error_processing(
    stream: list[ErrorSignature],
    window_size: int = 100
) -> dict:
    """
    Process error stream, tracking cache hit rate over time.

    Returns metrics about cache performance.
    """
    registry = FixRegistry()

    metrics = {
        "total_errors": len(stream),
        "cache_hits": 0,
        "cache_misses": 0,
        "hit_rate_over_time": [],  # (index, cumulative_hit_rate)
        "window_hit_rates": [],    # Hit rate in sliding windows
    }

    window_hits = []

    for i, error in enumerate(stream):
        # Try to lookup fix
        fix = registry.lookup(error)

        if fix is not None:
            # Cache hit - we already know how to fix this
            metrics["cache_hits"] += 1
            registry.report_success(error)
            window_hits.append(1)
        else:
            # Cache miss - need to "discover" fix and register
            metrics["cache_misses"] += 1
            new_fix = generate_fix_for_error(error)
            registry.register(error, new_fix)
            window_hits.append(0)

        # Record cumulative hit rate
        if (i + 1) % (len(stream) // 20) == 0 or i == len(stream) - 1:
            cumulative_rate = metrics["cache_hits"] / (i + 1)
            metrics["hit_rate_over_time"].append((i + 1, cumulative_rate))

        # Record window hit rate
        if len(window_hits) >= window_size:
            window_rate = sum(window_hits[-window_size:]) / window_size
            if len(metrics["window_hit_rates"]) == 0 or \
               len(window_hits) % (window_size // 2) == 0:
                metrics["window_hit_rates"].append((i + 1, window_rate))

    return metrics


def print_results(metrics: dict):
    """Print detailed results."""
    print("\n" + "="*70)
    print("RESULTS: Repeated Error Elimination")
    print("="*70)

    total = metrics["total_errors"]
    hits = metrics["cache_hits"]
    misses = metrics["cache_misses"]

    print(f"\nOverall Statistics:")
    print(f"  Total errors processed: {total:,}")
    print(f"  Cache hits:             {hits:,} ({100*hits/total:.1f}%)")
    print(f"  Cache misses:           {misses:,} ({100*misses/total:.1f}%)")
    print(f"  Final hit rate:         {100*hits/total:.1f}%")

    print(f"\nCumulative Hit Rate Over Time:")
    print(f"  {'Errors Processed':>18} | {'Hit Rate':>10}")
    print(f"  {'-'*18}-+-{'-'*10}")
    for idx, rate in metrics["hit_rate_over_time"]:
        print(f"  {idx:>18,} | {100*rate:>9.1f}%")

    if metrics["window_hit_rates"]:
        print(f"\nSliding Window Hit Rate:")
        print(f"  {'Position':>12} | {'Window Hit Rate':>16}")
        print(f"  {'-'*12}-+-{'-'*16}")
        for idx, rate in metrics["window_hit_rates"][-10:]:  # Last 10 windows
            print(f"  {idx:>12,} | {100*rate:>15.1f}%")

    print("\n" + "-"*70)
    print("Interpretation:")
    print("  - Early errors have low hit rate (learning phase)")
    print("  - Hit rate should climb rapidly as patterns repeat")
    print("  - Steady-state hit rate indicates error recurrence rate")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark repeated error elimination")
    parser.add_argument(
        "--length",
        type=int,
        default=1000,
        help="Number of errors in stream"
    )
    parser.add_argument(
        "--unique",
        type=int,
        default=50,
        help="Number of unique error patterns"
    )
    parser.add_argument(
        "--repeat-prob",
        type=float,
        default=0.7,
        help="Probability of repeating a seen error"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=100,
        help="Sliding window size for hit rate"
    )
    args = parser.parse_args()

    print("="*70)
    print("Experiment 2: Repeated Error Elimination")
    print("="*70)
    print(f"Stream length:    {args.length:,}")
    print(f"Unique patterns:  {args.unique}")
    print(f"Repeat prob:      {args.repeat_prob}")
    print(f"Window size:      {args.window}")

    print("\nGenerating error stream...")
    stream = generate_error_stream(
        length=args.length,
        num_unique_errors=args.unique,
        repeat_probability=args.repeat_prob
    )

    print("Processing errors...")
    metrics = simulate_error_processing(stream, window_size=args.window)

    print_results(metrics)


if __name__ == "__main__":
    main()
