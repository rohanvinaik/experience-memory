"""Tests for Error Taxonomy."""

import sys
import os
import pytest

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import (
    ErrorSeverity,
    ErrorType,
    ErrorSignature,
    classify_severity,
    create_signature,
)

def test_classify_severity():
    assert classify_severity(0.95) == ErrorSeverity.CATASTROPHIC
    assert classify_severity(0.6) == ErrorSeverity.MAJOR
    assert classify_severity(0.2) == ErrorSeverity.MODERATE
    assert classify_severity(0.05) == ErrorSeverity.MINOR

def test_signature_hashing():
    sig1 = create_signature(
        error_type=ErrorType.REMOVAL,
        context="test",
        affected_categories=["a", "b"],
        delta=0.6
    )
    
    sig2 = create_signature(
        error_type=ErrorType.REMOVAL,
        context="test",
        affected_categories=["b", "a"], # Different order
        delta=0.8 # Different delta
    )
    
    # Hashes should be identical despite different delta and list order
    assert sig1.hash() == sig2.hash()
