"""Background processor for continuous experience learning."""

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Dict, Any, List

from .dream_generator import DreamGenerator
from .fix_registry import FixRegistry

class ProcessorState(Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"

@dataclass
class ProcessorConfig:
    poll_interval_seconds: float = 30.0
    batch_size: int = 10
    max_items_per_cycle: int = 50
    epoch_interval_seconds: float = 3600.0
    min_item_age_seconds: float = 5.0

class BackgroundProcessor:
    """Async daemon for processing new items."""

    def __init__(self,
                 config: Optional[ProcessorConfig] = None,
                 dream_generator: Optional[DreamGenerator] = None,
                 fix_registry: Optional[FixRegistry] = None,
                 item_source: Optional[Callable[[], List[Dict]]] = None):
        self.config = config or ProcessorConfig()
        self.dream_generator = dream_generator
        self.fix_registry = fix_registry
        self.item_source = item_source
        self.state = ProcessorState.STOPPED
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        """Start background processing."""
        if self.state == ProcessorState.RUNNING:
            return
        
        self.state = ProcessorState.RUNNING
        self._stop_event.clear()
        
        # Start the processing loop
        asyncio.create_task(self._process_loop())

    async def stop(self) -> None:
        """Stop background processing."""
        self.state = ProcessorState.STOPPED
        self._stop_event.set()

    async def pause(self) -> None:
        """Pause processing."""
        if self.state == ProcessorState.RUNNING:
            self.state = ProcessorState.PAUSED

    async def resume(self) -> None:
        """Resume processing."""
        if self.state == ProcessorState.PAUSED:
            self.state = ProcessorState.RUNNING

    async def _process_loop(self) -> None:
        """Main processing loop."""
        while not self._stop_event.is_set():
            if self.state != ProcessorState.RUNNING:
                await asyncio.sleep(1)
                continue
                
            try:
                # 1. Advance epoch periodically (simplified, would check time)
                # self.fix_registry.advance_epoch()
                
                # 2. Fetch items if source provided
                if self.item_source and self.dream_generator:
                    items = self.item_source()
                    for item in items:
                        # Dream on each item
                        self.dream_generator.dream(item)
                        
            except Exception as e:
                # Log error
                print(f"Error in background processor: {e}")
                
            await asyncio.sleep(self.config.poll_interval_seconds)

    def get_metrics(self) -> Dict[str, Any]:
        """Return processing metrics."""
        return {
            "state": self.state.value,
            "epoch": self.fix_registry.decay_manager.current_epoch if self.fix_registry else 0
        }
