#!/usr/bin/env python3
"""
Experiment 1: Lookup Latency at Scale

Question: Does O(1) hold as registry grows?

This experiment validates the core claim that fix lookup is constant time
regardless of registry size.
"""

import argparse
import random
import time

from experience_memory import (
    FixRegistry,
    ErrorSignature,
    Fix,
    ErrorSeverity,
    ErrorType,
    FixType,
)


def generate_random_signature(i: int) -> ErrorSignature:
    """Generate a unique error signature."""
    severities = list(ErrorSeverity)
    error_types = list(ErrorType)
    contexts = [f"context_{i % 100}", f"task_{i % 50}", f"domain_{i % 25}"]
    categories = [["spatial"], ["temporal"], ["cognitive"], ["physical", "spatial"]]

    return ErrorSignature(
        severity=severities[i % len(severities)],
        error_type=error_types[i % len(error_types)],
        context=contexts[i % len(contexts)],
        affected_categories=categories[i % len(categories)],
        delta=random.uniform(0.1, 0.9),
    )


def generate_random_fix() -> Fix:
    """Generate a placeholder fix."""
    return Fix(
        fix_type=FixType.ADD_ELEMENTS,
        elements_to_add={"element": random.random()},
        elements_to_remove=[],
        value_adjustments={},
        category_moves={},
        definition_supplement=None,
        decomposition=None,
    )


def benchmark_lookup_latency(sizes: list[int], lookups_per_size: int = 10000):
    """
    Measure lookup latency at different registry sizes.

    Returns dict mapping size -> (mean_latency_ns, std_latency_ns)
    """
    results = {}

    for size in sizes:
        print(f"\n{'='*60}")
        print(f"Registry size: {size:,}")
        print('='*60)

        # Build registry
        registry = FixRegistry()
        signatures = []

        print(f"  Registering {size:,} fixes...")
        start = time.perf_counter()
        for i in range(size):
            sig = generate_random_signature(i)
            fix = generate_random_fix()
            registry.register(sig, fix)
            signatures.append(sig)
        register_time = time.perf_counter() - start
        print(f"  Registration time: {register_time:.3f}s ({size/register_time:.0f} ops/sec)")

        # Benchmark lookups
        print(f"  Running {lookups_per_size:,} lookups...")
        latencies = []

        for _ in range(lookups_per_size):
            sig = random.choice(signatures)

            start = time.perf_counter_ns()
            result = registry.lookup(sig)
            end = time.perf_counter_ns()

            latencies.append(end - start)
            assert result is not None, "Lookup should find registered fix"

        # Statistics
        mean_ns = sum(latencies) / len(latencies)
        variance = sum((x - mean_ns) ** 2 for x in latencies) / len(latencies)
        std_ns = variance ** 0.5
        min_ns = min(latencies)
        max_ns = max(latencies)
        p50 = sorted(latencies)[len(latencies) // 2]
        p99 = sorted(latencies)[int(len(latencies) * 0.99)]

        results[size] = {
            "mean_ns": mean_ns,
            "std_ns": std_ns,
            "min_ns": min_ns,
            "max_ns": max_ns,
            "p50_ns": p50,
            "p99_ns": p99,
        }

        print(f"\n  Results:")
        print(f"    Mean latency:  {mean_ns:,.0f} ns ({mean_ns/1000:.2f} µs)")
        print(f"    Std deviation: {std_ns:,.0f} ns")
        print(f"    Min:           {min_ns:,.0f} ns")
        print(f"    Max:           {max_ns:,.0f} ns")
        print(f"    P50:           {p50:,.0f} ns")
        print(f"    P99:           {p99:,.0f} ns")

    return results


def print_summary(results: dict):
    """Print summary table showing O(1) behavior."""
    print("\n" + "="*70)
    print("SUMMARY: Lookup Latency vs Registry Size")
    print("="*70)
    print(f"{'Size':>12} | {'Mean (µs)':>12} | {'P99 (µs)':>12} | {'Ratio vs 100':>14}")
    print("-"*70)

    baseline = results.get(100, results[min(results.keys())])["mean_ns"]

    for size in sorted(results.keys()):
        r = results[size]
        ratio = r["mean_ns"] / baseline
        print(f"{size:>12,} | {r['mean_ns']/1000:>12.2f} | {r['p99_ns']/1000:>12.2f} | {ratio:>14.2f}x")

    print("-"*70)
    print("\nConclusion: If O(1) holds, 'Ratio vs 100' should stay close to 1.0")
    print("            regardless of registry size.\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark lookup latency at scale")
    parser.add_argument(
        "--sizes",
        type=str,
        default="100,1000,10000,100000",
        help="Comma-separated registry sizes to test"
    )
    parser.add_argument(
        "--lookups",
        type=int,
        default=10000,
        help="Number of lookups per size"
    )
    args = parser.parse_args()

    sizes = [int(s.strip()) for s in args.sizes.split(",")]

    print("="*70)
    print("Experiment 1: Lookup Latency at Scale")
    print("="*70)
    print(f"Testing sizes: {sizes}")
    print(f"Lookups per size: {args.lookups:,}")

    results = benchmark_lookup_latency(sizes, args.lookups)
    print_summary(results)


if __name__ == "__main__":
    main()
