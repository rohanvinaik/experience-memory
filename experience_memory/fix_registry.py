from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Protocol
import hashlib
import json

from .error_taxonomy import ErrorSignature
from .decay import DecayManager, DecayConfig, Decayable

# FixType enum (was FixSuggestionType)
class FixType(Enum):
    ADD_ELEMENTS = "add"           # Was ADD_PRIMITIVES
    REMOVE_ELEMENTS = "remove"     # Was REMOVE_PRIMITIVES
    ADJUST_VALUES = "adjust"       # Was ADJUST_INTENSITIES
    SWAP_ELEMENTS = "swap"         # Was SWAP_PRIMITIVES
    RESTORE_CATEGORY = "restore"   # Was RESTORE_BANK
    INVERT_RELATIONSHIP = "invert" # Keep
    COMPLETE_DEFINITION = "complete"  # Keep
    DECOMPOSE = "decompose"        # Was DECOMPOSE_CONCEPT

# Fix dataclass (was FixSuggestion)
@dataclass
class Fix:
    fix_type: FixType
    elements_to_add: Dict[str, float]     # Was primitives_to_add
    elements_to_remove: List[str]         # Was primitives_to_remove
    value_adjustments: Dict[str, float]   # Was intensity_adjustments
    category_moves: Dict[str, str]        # Was bank_moves
    definition_supplement: Optional[str]
    decomposition: Optional[List[str]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fix_type": self.fix_type.value,
            "elements_to_add": self.elements_to_add,
            "elements_to_remove": self.elements_to_remove,
            "value_adjustments": self.value_adjustments,
            "category_moves": self.category_moves,
            "definition_supplement": self.definition_supplement,
            "decomposition": self.decomposition,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Fix":
        return cls(
            fix_type=FixType(data["fix_type"]),
            elements_to_add=data["elements_to_add"],
            elements_to_remove=data["elements_to_remove"],
            value_adjustments=data["value_adjustments"],
            category_moves=data["category_moves"],
            definition_supplement=data.get("definition_supplement"),
            decomposition=data.get("decomposition"),
        )

# FixEntry dataclass (was AntibodyEntry)
@dataclass
class FixEntry:
    error_signature_hash: str      # 16-char SHA256
    fix: Fix
    strength: float = 1.0          # Decay tracking
    success_count: int = 0
    failure_count: int = 0
    creation_epoch: int = 0
    last_success_epoch: Optional[int] = None
    last_access_epoch: Optional[int] = None

# FixRegistry class (was AntibodyRegistry)
class FixRegistry:
    """O(1) error-to-fix mapping with temporal decay."""

    def __init__(self, db_path: Optional[str] = None, decay_config: Optional[DecayConfig] = None):
        """Initialize with optional SQLite persistence."""
        self._entries: Dict[str, FixEntry] = {}
        self.decay_manager = DecayManager(config=decay_config)
        self.db_path = db_path
        # TODO: Implement persistence loading

    def register(self, signature: ErrorSignature, fix: Fix) -> None:
        """Register a fix for an error signature."""
        error_hash = signature.hash()
        entry = FixEntry(
            error_signature_hash=error_hash,
            fix=fix,
            creation_epoch=self.decay_manager.current_epoch,
            last_access_epoch=self.decay_manager.current_epoch
        )
        self._entries[error_hash] = entry

    def lookup(self, signature: ErrorSignature) -> Optional[Fix]:
        """O(1) lookup of fix for error signature."""
        error_hash = signature.hash()
        entry = self._entries.get(error_hash)
        if entry:
            entry.last_access_epoch = self.decay_manager.current_epoch
            return entry.fix
        return None

    def _lookup_entry(self, signature: ErrorSignature) -> Optional[FixEntry]:
        error_hash = signature.hash()
        return self._entries.get(error_hash)

    def report_success(self, signature: ErrorSignature) -> None:
        """Strengthen fix after successful application."""
        entry = self._lookup_entry(signature)
        if entry:
            entry.strength = self.decay_manager.reinforce(entry)
            entry.success_count += 1
            entry.last_success_epoch = self.decay_manager.current_epoch
            entry.last_access_epoch = self.decay_manager.current_epoch

    def report_failure(self, signature: ErrorSignature) -> None:
        """Weaken fix after failed application."""
        error_hash = signature.hash()
        entry = self._entries.get(error_hash)
        if entry:
            entry.strength = self.decay_manager.penalize(entry)
            entry.failure_count += 1
            entry.last_access_epoch = self.decay_manager.current_epoch
            if self.decay_manager.should_prune(entry):
                self._prune_entry(signature)

    def _prune_entry(self, signature: ErrorSignature) -> None:
        error_hash = signature.hash()
        if error_hash in self._entries:
            del self._entries[error_hash]

    def advance_epoch(self) -> int:
        """Apply decay to all entries, prune weak ones. Returns pruned count."""
        self.decay_manager.advance_epoch()
        pruned_count = 0
        to_prune = []
        
        for error_hash, entry in self._entries.items():
            entry.strength = self.decay_manager.apply_decay(entry)
            if self.decay_manager.should_prune(entry):
                to_prune.append(error_hash)
        
        for error_hash in to_prune:
            del self._entries[error_hash]
            pruned_count += 1
            
        return pruned_count

    def get_stats(self) -> Dict[str, Any]:
        """Return registry statistics."""
        return {
            "total_entries": len(self._entries),
            "current_epoch": self.decay_manager.current_epoch,
        }
