"""Experience Memory: Turn failures into O(1) fixes without retraining."""

from .error_taxonomy import (
    ErrorSeverity,
    ErrorType,
    ErrorSignature,
    classify_severity,
    create_signature,
)

from .fix_registry import (
    FixType,
    Fix,
    FixEntry,
    FixRegistry,
)

from .dream_generator import (
    SyntheticExample,
    DreamGenerator,
)

from .decay import (
    DecayConfig,
    DecayManager,
)

from .background_processor import (
    ProcessorConfig,
    ProcessorState,
    BackgroundProcessor,
)

__version__ = "0.1.0"

__all__ = [
    # Error taxonomy
    "ErrorSeverity",
    "ErrorType",
    "ErrorSignature",
    "classify_severity",
    "create_signature",
    # Fix registry
    "FixType",
    "Fix",
    "FixEntry",
    "FixRegistry",
    # Dream generation
    "SyntheticExample",
    "DreamGenerator",
    # Decay
    "DecayConfig",
    "DecayManager",
    # Background Processor
    "ProcessorConfig",
    "ProcessorState",
    "BackgroundProcessor",
]
