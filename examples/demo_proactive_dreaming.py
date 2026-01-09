"""Demo: Generate synthetic edge cases before deployment."""

import sys
import os

# Add parent directory to path so we can import the local package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import (
    DreamGenerator,
    FixRegistry,
)

def main():
    # Initialize with auto-registration
    registry = FixRegistry()
    dreamer = DreamGenerator(registry=registry)

    # A concept we've learned
    concept = {
        "term": "containment",
        "definition": "Keeping something within boundaries",
        "activations": {
            "boundary": 1.0,
            "inside": 0.9,
            "outside": 0.3,
            "contain": 0.85,
        },
        "categories": ["spatial", "relational"],
    }

    # Generate synthetic failures
    synthetic_examples = dreamer.dream(
        concept=concept,
        strategies=["removal", "substitution", "intensity"],
        variants_per_strategy=3,
    )

    print(f"Generated {len(synthetic_examples)} synthetic examples:")
    for i, example in enumerate(synthetic_examples):
        print(f"\n--- Example {i+1} ---")
        print(f"Strategy: {example.corruption_strategy}")
        print(f"Error type: {example.error_signature.error_type.value}")
        print(f"Severity: {example.error_signature.severity.value}")
        print(f"Fix type: {example.fix.fix_type.value}")

    # All examples auto-registered
    stats = registry.get_stats()
    print(f"\nRegistry now has {stats['total_entries']} pre-registered fixes")
    print("System is proactively protected against these error types!")

if __name__ == "__main__":
    main()
