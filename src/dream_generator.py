from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import random
import copy

from .error_taxonomy import ErrorSignature, ErrorType, create_signature
from .fix_registry import Fix, FixType, FixRegistry

# DreamPair → SyntheticExample
@dataclass
class SyntheticExample:
    original: Dict[str, float]      # Original state
    corrupted: Dict[str, float]     # Corrupted state
    error_signature: ErrorSignature
    fix: Fix
    corruption_strategy: str        # Which strategy generated this

# DreamGenerator (consolidates 4 pathway dreamers)
class DreamGenerator:
    """Generate synthetic error examples for proactive learning."""

    STRATEGIES = [
        "removal",      # Remove 1-3 elements
        "substitution", # Swap with neighbors
        "intensity",    # Perturb values (0.3x, 0.5x, 0.7x)
        "structural",   # Move to wrong category
        "inversion",    # Flip relationships
        "partial",      # Truncate definitions
        "decomposition" # Remove components
    ]
    
    # Intensity perturbation scales
    INTENSITY_SCALES = [0.3, 0.5, 0.7]

    # Removal counts
    REMOVAL_COUNTS = [1, 2, 3]

    # Relationship inversions (from HebbianDreamer)
    OPPOSITE_PAIRS = {
        "good": "bad", "bad": "good",
        "true": "false", "false": "true",
        "more": "less", "less": "more",
        "all": "none", "none": "all",
        "same": "different", "different": "same",
        "help": "harm", "harm": "help",
        "give": "take", "take": "give",
        "before": "after", "after": "before",
        "up": "down", "down": "up",
        "inside": "outside", "outside": "inside",
    }

    # Strength perturbation multipliers (from HebbianDreamer)
    STRENGTH_MULTIPLIERS = [0.1, 0.5, 2.0, 5.0]

    # Partial definition ratios
    TRUNCATION_RATIOS = [0.3, 0.5, 0.7]

    # Component removal ratios
    COMPONENT_REMOVAL_RATIOS = [0.2, 0.4, 0.6]

    def __init__(self, registry: Optional[FixRegistry] = None):
        """Initialize with optional registry for auto-registration."""
        self.registry = registry

    def dream(self,
              concept: Dict[str, Any],
              strategies: Optional[List[str]] = None,
              variants_per_strategy: int = 3) -> List[SyntheticExample]:
        """Generate synthetic examples from a learned concept."""
        strategies = strategies or self.STRATEGIES
        examples = []
        
        # Helper to simplify getting activations
        activations = concept.get("activations", {})
        
        for strategy in strategies:
            for _ in range(variants_per_strategy):
                if strategy == "removal":
                    example = self._removal_corruption(concept)
                elif strategy == "substitution":
                     # Needs neighbors, skipping for simple implementation without extra data
                     continue 
                elif strategy == "intensity":
                    scale = random.choice(self.INTENSITY_SCALES)
                    example = self._intensity_corruption(concept, scale)
                elif strategy == "structural":
                    # Needs categories, simplified
                    continue
                elif strategy == "inversion":
                    example = self._inversion_corruption(concept)
                # ... implement other strategies
                else: 
                     continue
                
                if example:
                    examples.append(example)
                    if self.registry:
                        self.registry.register(example.error_signature, example.fix)
                        
        return examples

    def _removal_corruption(self, concept: Dict[str, Any]) -> Optional[SyntheticExample]:
        """Remove 1-3 random elements."""
        activations = concept.get("activations", {})
        if not activations:
            return None
            
        keys = list(activations.keys())
        num_remove = min(len(keys), random.choice(self.REMOVAL_COUNTS))
        
        to_remove = random.sample(keys, num_remove)
        corrupted = activations.copy()
        removed_elements = []
        
        for key in to_remove:
            del corrupted[key]
            removed_elements.append(key)
            
        # Calculate delta (simplified)
        delta = len(removed_elements) / len(activations) if activations else 0.0
        
        sig = create_signature(
            error_type=ErrorType.REMOVAL,
            context="dreaming",
            affected_categories=concept.get("categories", []),
            delta=delta
        )
        
        fix = Fix(
            fix_type=FixType.ADD_ELEMENTS,
            elements_to_add={k: activations[k] for k in removed_elements},
            elements_to_remove=[],
            value_adjustments={},
            category_moves={},
            definition_supplement=None,
            decomposition=None
        )
        
        return SyntheticExample(
            original=activations,
            corrupted=corrupted,
            error_signature=sig,
            fix=fix,
            corruption_strategy="removal"
        )

    def _intensity_corruption(self, concept: Dict[str, Any],
                               scale: float) -> Optional[SyntheticExample]:
        """Scale values by 0.3x, 0.5x, or 0.7x."""
        activations = concept.get("activations", {})
        if not activations:
            return None
            
        corrupted = {k: v * scale for k, v in activations.items()}
        
        # Calculate delta (simplified difference)
        delta = abs(1.0 - scale)
        
        sig = create_signature(
            error_type=ErrorType.INTENSITY_SHIFT,
            context="dreaming",
            affected_categories=concept.get("categories", []),
            delta=delta
        )
        
        fix = Fix(
            fix_type=FixType.ADJUST_VALUES,
            elements_to_add={},
            elements_to_remove=[],
            # Invert the scale to fix
            value_adjustments={k: 1.0/scale for k in activations}, 
            category_moves={},
            definition_supplement=None,
            decomposition=None
        )
        
        return SyntheticExample(
            original=activations,
            corrupted=corrupted,
            error_signature=sig,
            fix=fix,
            corruption_strategy="intensity"
        )

    def _inversion_corruption(self, concept: Dict[str, Any]) -> Optional[SyntheticExample]:
        """Flip relationship polarities."""
        # Check against string representation or keys
        # This one is tricky without graph struct, simplified for demo
        return None 
