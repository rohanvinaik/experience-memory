from enum import Enum
from dataclasses import dataclass
from typing import List
import hashlib

# ErrorSeverity enum
class ErrorSeverity(Enum):
    CATASTROPHIC = "catastrophic"  # delta > 0.9
    MAJOR = "major"                # delta > 0.5
    MODERATE = "moderate"          # delta > 0.1
    MINOR = "minor"                # delta <= 0.1

# CorruptionType enum → RENAME to ErrorType
class ErrorType(Enum):
    REMOVAL = "removal"              # Missing elements
    SUBSTITUTION = "substitution"    # Wrong element used
    INTENSITY_SHIFT = "intensity"    # Value perturbation
    STRUCTURAL_SWAP = "structural"   # Wrong container/category
    RELATIONSHIP_INVERSION = "inversion"  # Opposite relationship
    PARTIAL_DEFINITION = "partial"   # Incomplete specification
    COMPONENT_REMOVAL = "decomposition"  # Missing sub-parts

# ErrorSignature dataclass
@dataclass
class ErrorSignature:
    severity: ErrorSeverity
    error_type: ErrorType
    context: str              # Replaces "pathway" - more generic
    affected_categories: List[str]  # Replaces "affected_banks"
    delta: float              # 0.0-1.0 change magnitude

    def hash(self) -> str:
        """Generate 16-char hash for O(1) lookup (delta excluded for grouping)."""
        key = f"{self.severity.value}:{self.error_type.value}:{self.context}:{sorted(self.affected_categories)}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

def classify_severity(delta: float) -> ErrorSeverity:
    """Classify error severity based on delta."""
    if delta > 0.9:
        return ErrorSeverity.CATASTROPHIC
    elif delta > 0.5:
        return ErrorSeverity.MAJOR
    elif delta > 0.1:
        return ErrorSeverity.MODERATE
    else:
        return ErrorSeverity.MINOR

def create_signature(error_type: ErrorType, context: str, affected_categories: List[str], delta: float) -> ErrorSignature:
    """Create an error signature."""
    severity = classify_severity(delta)
    return ErrorSignature(
        severity=severity,
        error_type=error_type,
        context=context,
        affected_categories=affected_categories,
        delta=delta,
    )
