#!/usr/bin/env python3
"""
Experiment 4: Memory Bounds Under Decay

Question: Does decay prevent unbounded growth?

This experiment simulates continuous operation where new fixes are registered
while decay runs, showing that memory usage reaches a steady state.
"""

import argparse
import random

from experience_memory import (
    FixRegistry,
    ErrorSignature,
    Fix,
    ErrorSeverity,
    ErrorType,
    FixType,
    DecayConfig,
)


def generate_signature(i: int) -> ErrorSignature:
    """Generate unique error signature."""
    return ErrorSignature(
        severity=list(ErrorSeverity)[i % 4],
        error_type=list(ErrorType)[i % 7],
        context=f"ctx_{i}",
        affected_categories=[["spatial"], ["temporal"], ["physical"]][i % 3],
        delta=random.uniform(0.1, 0.9),
    )


def generate_fix() -> Fix:
    """Generate placeholder fix."""
    return Fix(
        fix_type=FixType.ADD_ELEMENTS,
        elements_to_add={"x": 1.0},
        elements_to_remove=[],
        value_adjustments={},
        category_moves={},
        definition_supplement=None,
        decomposition=None,
    )


def simulate_continuous_operation(
    epochs: int,
    registrations_per_epoch: int,
    decay_config: DecayConfig,
    reinforce_fraction: float = 0.0
) -> dict:
    """
    Simulate continuous operation with registration and decay.

    Args:
        epochs: Number of epochs to simulate
        registrations_per_epoch: New fixes registered each epoch
        decay_config: Decay configuration
        reinforce_fraction: Fraction of existing entries to reinforce each epoch
    """
    registry = FixRegistry(decay_config=decay_config)

    metrics = {
        "epoch": [],
        "registry_size": [],
        "pruned_this_epoch": [],
        "registered_this_epoch": [],
        "reinforced_this_epoch": [],
    }

    all_signatures = []
    sig_counter = 0

    for epoch in range(epochs):
        # Register new fixes
        registered = 0
        for _ in range(registrations_per_epoch):
            sig = generate_signature(sig_counter)
            fix = generate_fix()
            registry.register(sig, fix)
            all_signatures.append(sig)
            sig_counter += 1
            registered += 1

        # Optionally reinforce some existing entries
        reinforced = 0
        if reinforce_fraction > 0 and all_signatures:
            num_reinforce = int(len(all_signatures) * reinforce_fraction)
            for sig in random.sample(all_signatures, min(num_reinforce, len(all_signatures))):
                if registry.lookup(sig) is not None:
                    registry.report_success(sig)
                    reinforced += 1

        # Apply decay
        pruned = registry.advance_epoch()

        # Record metrics
        stats = registry.get_stats()
        metrics["epoch"].append(epoch)
        metrics["registry_size"].append(stats["total_entries"])
        metrics["pruned_this_epoch"].append(pruned)
        metrics["registered_this_epoch"].append(registered)
        metrics["reinforced_this_epoch"].append(reinforced)

        # Clean up pruned signatures from tracking
        all_signatures = [s for s in all_signatures if registry.lookup(s) is not None]

    return metrics


def print_results(metrics: dict, decay_config: DecayConfig):
    """Print analysis of memory bounds."""
    print("\n" + "="*70)
    print("RESULTS: Memory Bounds Under Decay")
    print("="*70)

    sizes = metrics["registry_size"]
    peak_size = max(sizes)
    final_size = sizes[-1]
    total_registered = sum(metrics["registered_this_epoch"])
    total_pruned = sum(metrics["pruned_this_epoch"])

    print(f"\nConfiguration:")
    print(f"  Exponential decay rate: {decay_config.exponential_rate}")
    print(f"  Linear decay rate:      {decay_config.linear_rate}")
    print(f"  Min strength threshold: {decay_config.min_strength}")

    print(f"\nOverall Statistics:")
    print(f"  Total epochs:           {len(metrics['epoch'])}")
    print(f"  Total registered:       {total_registered:,}")
    print(f"  Total pruned:           {total_pruned:,}")
    print(f"  Peak registry size:     {peak_size:,}")
    print(f"  Final registry size:    {final_size:,}")
    print(f"  Retention rate:         {100*final_size/total_registered:.1f}%")

    # Find steady state
    last_quarter = sizes[-(len(sizes)//4):]
    avg_last_quarter = sum(last_quarter) / len(last_quarter)
    std_last_quarter = (sum((x - avg_last_quarter)**2 for x in last_quarter) / len(last_quarter)) ** 0.5

    print(f"\nSteady State Analysis (last 25% of epochs):")
    print(f"  Average size:           {avg_last_quarter:.1f}")
    print(f"  Std deviation:          {std_last_quarter:.1f}")
    print(f"  Coefficient of var:     {100*std_last_quarter/avg_last_quarter:.1f}%")

    # Print trajectory
    print(f"\nRegistry Size Over Time:")
    print(f"  {'Epoch':>8} | {'Size':>10} | {'Registered':>12} | {'Pruned':>10}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*12}-+-{'-'*10}")

    step = max(1, len(sizes) // 20)
    for i in range(0, len(sizes), step):
        print(f"  {metrics['epoch'][i]:>8} | {metrics['registry_size'][i]:>10,} | "
              f"{metrics['registered_this_epoch'][i]:>12} | {metrics['pruned_this_epoch'][i]:>10}")
    # Always show last
    if (len(sizes) - 1) % step != 0:
        i = len(sizes) - 1
        print(f"  {metrics['epoch'][i]:>8} | {metrics['registry_size'][i]:>10,} | "
              f"{metrics['registered_this_epoch'][i]:>12} | {metrics['pruned_this_epoch'][i]:>10}")

    print("\n" + "-"*70)
    print("Interpretation:")
    print("  - If size stabilizes, decay is bounding memory growth")
    print("  - Steady state size depends on decay rate vs registration rate")
    print("  - Low CoV in last quarter indicates stable steady state")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark memory bounds under decay")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("--rate", type=int, default=10, help="Registrations per epoch")
    parser.add_argument("--exp-decay", type=float, default=0.97, help="Exponential decay rate")
    parser.add_argument("--lin-decay", type=float, default=0.001, help="Linear decay rate")
    parser.add_argument("--threshold", type=float, default=0.01, help="Pruning threshold")
    parser.add_argument("--reinforce", type=float, default=0.0,
                        help="Fraction of entries to reinforce each epoch")
    args = parser.parse_args()

    config = DecayConfig(
        exponential_rate=args.exp_decay,
        linear_rate=args.lin_decay,
        min_strength=args.threshold,
    )

    print("="*70)
    print("Experiment 4: Memory Bounds Under Decay")
    print("="*70)
    print(f"Epochs:               {args.epochs}")
    print(f"Registrations/epoch:  {args.rate}")
    print(f"Exponential decay:    {args.exp_decay}")
    print(f"Linear decay:         {args.lin_decay}")
    print(f"Prune threshold:      {args.threshold}")
    print(f"Reinforce fraction:   {args.reinforce}")

    print("\nRunning simulation...")
    metrics = simulate_continuous_operation(
        epochs=args.epochs,
        registrations_per_epoch=args.rate,
        decay_config=config,
        reinforce_fraction=args.reinforce,
    )

    print_results(metrics, config)


if __name__ == "__main__":
    main()
