"""Tests for FixRegistry."""

import pytest
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import (
    FixRegistry,
    ErrorSignature,
    ErrorSeverity,
    ErrorType,
    Fix,
    FixType,
)

class TestFixRegistry:

    def test_register_and_lookup(self):
        """Test basic registration and O(1) lookup."""
        registry = FixRegistry()

        error = ErrorSignature(
            severity=ErrorSeverity.MAJOR,
            error_type=ErrorType.REMOVAL,
            context="test",
            affected_categories=["spatial"],
            delta=0.6,
        )

        fix = Fix(
            fix_type=FixType.ADD_ELEMENTS,
            elements_to_add={"boundary": 1.0},
            elements_to_remove=[],
            value_adjustments={},
            category_moves={},
            definition_supplement=None,
            decomposition=None,
        )

        registry.register(error, fix)

        # Lookup should work
        result = registry.lookup(error)
        assert result is not None
        assert result.fix_type == FixType.ADD_ELEMENTS
        assert result.elements_to_add == {"boundary": 1.0}

    def test_lookup_ignores_delta(self):
        """Test that delta is excluded from hash (groups similar errors)."""
        registry = FixRegistry()

        error1 = ErrorSignature(
            severity=ErrorSeverity.MAJOR,
            error_type=ErrorType.REMOVAL,
            context="test",
            affected_categories=["spatial"],
            delta=0.6,  # Different delta
        )

        error2 = ErrorSignature(
            severity=ErrorSeverity.MAJOR,
            error_type=ErrorType.REMOVAL,
            context="test",
            affected_categories=["spatial"],
            delta=0.8,  # Different delta
        )

        fix = Fix(
            fix_type=FixType.ADD_ELEMENTS,
            elements_to_add={"boundary": 1.0},
            elements_to_remove=[],
            value_adjustments={},
            category_moves={},
            definition_supplement=None,
            decomposition=None,
        )

        registry.register(error1, fix)

        # Should find the same fix
        result = registry.lookup(error2)
        assert result is not None

    def test_success_reinforcement(self):
        """Test that success increases strength."""
        registry = FixRegistry()

        error = ErrorSignature(
            severity=ErrorSeverity.MINOR,
            error_type=ErrorType.INTENSITY_SHIFT,
            context="test",
            affected_categories=[],
            delta=0.05
        )
        
        fix = Fix(
            fix_type=FixType.ADJUST_VALUES,
            elements_to_add={},
            elements_to_remove=[],
            value_adjustments={},
            category_moves={},
            definition_supplement=None,
            decomposition=None,
        )
        
        registry.register(error, fix)
        entry = registry._lookup_entry(error)
        entry.strength = 0.5 # Lower strength to allow increase
        initial_strength = entry.strength
        
        registry.report_success(error)
        assert entry.strength > initial_strength
