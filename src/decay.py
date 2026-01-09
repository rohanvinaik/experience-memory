"""Temporal decay mechanisms for experience memory."""

from dataclasses import dataclass
from typing import Protocol, TypeVar, Optional

T = TypeVar('T')

class Decayable(Protocol):
    """Protocol for objects with decay."""
    strength: float
    last_access_epoch: Optional[int]

@dataclass
class DecayConfig:
    """Configuration for decay behavior."""
    exponential_rate: float = 0.97    # Per-epoch multiplier
    linear_rate: float = 0.001        # Per-epoch subtraction
    min_strength: float = 0.01        # Pruning threshold
    success_boost: float = 0.05       # Reinforcement amount
    failure_penalty: float = 0.10     # Penalty amount

class DecayManager:
    """Manage temporal decay for entries."""

    def __init__(self, config: Optional[DecayConfig] = None):
        self.config = config or DecayConfig()
        self._current_epoch = 0

    def apply_decay(self, entry: Decayable) -> float:
        """Apply decay formula: strength = strength * exp_rate - linear_rate"""
        new_strength = entry.strength * self.config.exponential_rate
        new_strength -= self.config.linear_rate
        return max(0.0, new_strength)

    def should_prune(self, entry: Decayable) -> bool:
        """Check if entry should be removed."""
        return entry.strength < self.config.min_strength

    def reinforce(self, entry: Decayable) -> float:
        """Strengthen after success."""
        return min(1.0, entry.strength + self.config.success_boost)

    def penalize(self, entry: Decayable) -> float:
        """Weaken after failure."""
        return max(0.0, entry.strength - self.config.failure_penalty)

    def advance_epoch(self) -> int:
        """Advance epoch counter."""
        self._current_epoch += 1
        return self._current_epoch

    @property
    def current_epoch(self) -> int:
        return self._current_epoch
