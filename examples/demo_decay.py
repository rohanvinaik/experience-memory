"""Demo: Temporal decay and memory management."""

import sys
import os

# Add parent directory to path so we can import the local package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import (
    FixRegistry,
    ErrorSignature,
    ErrorSeverity,
    ErrorType,
    Fix,
    FixType,
    DecayConfig,
)

def main():
    # Custom decay configuration
    config = DecayConfig(
        exponential_rate=0.97,
        linear_rate=0.001,
        min_strength=0.01,
        success_boost=0.05,
        failure_penalty=0.10,
    )

    registry = FixRegistry(decay_config=config)

    # Register a fix
    error = ErrorSignature(
        severity=ErrorSeverity.MODERATE,
        error_type=ErrorType.SUBSTITUTION,
        context="color_recognition",
        affected_categories=["physical"],
        delta=0.35,
    )

    fix = Fix(
        fix_type=FixType.SWAP_ELEMENTS,
        elements_to_add={},
        elements_to_remove=["red"],
        value_adjustments={},
        category_moves={},
        definition_supplement=None,
        decomposition=None,
    )

    registry.register(error, fix)
    print(f"Initial strength: 1.0")

    # Simulate epochs passing without use
    for epoch in range(1, 101):
        pruned = registry.advance_epoch()
        entry = registry._lookup_entry(error)

        if entry is None:
            print(f"Epoch {epoch}: Entry PRUNED (strength fell below {config.min_strength})")
            break
        elif epoch % 20 == 0:
            print(f"Epoch {epoch}: strength = {entry.strength:.4f}")

    # With reinforcement
    print("\n--- With periodic reinforcement ---")
    registry2 = FixRegistry(decay_config=config)
    registry2.register(error, fix)

    for epoch in range(1, 101):
        pruned = registry2.advance_epoch()

        # Reinforce every 10 epochs
        if epoch % 10 == 0:
            registry2.report_success(error)

        entry = registry2._lookup_entry(error)
        if epoch % 20 == 0 and entry:
            print(f"Epoch {epoch}: strength = {entry.strength:.4f}")

    print("\nEntry survived 100 epochs due to periodic reinforcement!")

if __name__ == "__main__":
    main()
