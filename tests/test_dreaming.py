"""Tests for Dream Generator."""

import sys
import os
import pytest
from unittest.mock import MagicMock

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import (
    DreamGenerator,
    FixRegistry,
    ErrorType,
)

def test_dream_generation():
    registry = MagicMock(spec=FixRegistry)
    dreamer = DreamGenerator(registry=registry)
    
    concept = {
        "term": "test_concept",
        "activations": {"a": 1.0, "b": 0.5},
        "categories": ["test_cat"]
    }
    
    # Test removal strategy
    examples = dreamer.dream(concept, strategies=["removal"], variants_per_strategy=5)
    
    assert len(examples) > 0
    for example in examples:
        assert example.corruption_strategy == "removal"
        assert example.error_signature.error_type == ErrorType.REMOVAL
        # Verify auto-registration called
        registry.register.assert_called()

def test_intensity_corruption():
    dreamer = DreamGenerator()
    concept = {
        "term": "test_concept",
        "activations": {"a": 1.0},
        "categories": ["test_cat"]
    }
    
    examples = dreamer.dream(concept, strategies=["intensity"], variants_per_strategy=1)
    assert len(examples) > 0
    assert examples[0].error_signature.error_type == ErrorType.INTENSITY_SHIFT
