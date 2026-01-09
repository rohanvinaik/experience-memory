"""Demo: Learn from a failure and apply O(1) fix lookup."""

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
)

def main():
    # Initialize registry (in-memory or persistent)
    registry = FixRegistry()

    # Simulate a failure: boundary was removed from containment task
    error = ErrorSignature(
        severity=ErrorSeverity.MAJOR,
        error_type=ErrorType.REMOVAL,
        context="containment_task",
        affected_categories=["spatial", "relational"],
        delta=0.65,
    )

    # Create the fix
    fix = Fix(
        fix_type=FixType.ADD_ELEMENTS,
        elements_to_add={"boundary": 1.0, "inside": 0.8},
        elements_to_remove=[],
        value_adjustments={},
        category_moves={},
        definition_supplement=None,
        decomposition=None,
    )

    # Register the fix
    registry.register(error, fix)
    print(f"Registered fix for error hash: {error.hash()}")

    # Later: Same error occurs
    same_error = ErrorSignature(
        severity=ErrorSeverity.MAJOR,
        error_type=ErrorType.REMOVAL,
        context="containment_task",
        affected_categories=["spatial", "relational"],
        delta=0.72,  # Different delta, same hash (delta excluded from hash)
    )

    # O(1) lookup
    retrieved_fix = registry.lookup(same_error)
    if retrieved_fix:
        print(f"Found fix: {retrieved_fix.fix_type.value}")
        print(f"Add elements: {retrieved_fix.elements_to_add}")

        # Report success after applying
        registry.report_success(same_error)
        print("Fix succeeded, strength reinforced")

    # Show stats
    stats = registry.get_stats()
    print(f"\nRegistry stats: {stats}")

if __name__ == "__main__":
    main()
