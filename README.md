# Experience Memory

Adapt from failures in O(1) time without gradient descent.

When the same error pattern recurs, apply the cached fix in ~1.6 μs instead of retraining. The registry scales: 500× more entries → only 1.22× slower.

---

## The Problem

Standard error remediation is expensive:

| Approach | Time to Fix | Compute Cost |
|----------|-------------|--------------|
| Full retraining | Hours-Days | GPU clusters |
| Fine-tuning | Minutes-Hours | Medium |
| Manual rules | Days (human time) | Low |
| **This system** | **~1.6 μs** | **None** |

---

## Key Results

### O(1) Lookup (Verified)

| Registry Size | Mean Latency | Ratio vs 100 |
|---------------|--------------|--------------|
| 100 | 1.33 μs | 1.00× |
| 1,000 | 1.37 μs | 1.03× |
| 10,000 | 1.41 μs | 1.06× |
| 50,000 | 1.63 μs | 1.22× |

### Error Elimination

| Metric | Value |
|--------|-------|
| Errors processed | 1,000 |
| Final cache hit rate | 97.0% |
| Steady-state hit rate | 100% |

After learning 30 unique patterns, the system achieves near-perfect cache hits.

### Memory Bounds

Despite 5,000 registrations over 500 epochs, decay maintains bounded memory at ~1,040 entries.

---

## How It Works

```
Error occurs → Classify by type/context → Hash to signature → O(1) lookup
  │
  ├── Cache hit? Apply cached fix
  │
  └── Cache miss? Generate fix, register signature → fix mapping
```

**Key design**: Hash excludes magnitude, so a "60% missing" error and "90% missing" error of the same type get the same fix.

---

## Quick Start

```bash
git clone https://github.com/rohanvinaik/experience-memory.git
cd experience-memory
pip install -e .
```

```python
from experience_memory import FixRegistry, ErrorSignature, ErrorSeverity, ErrorType

registry = FixRegistry()

error = ErrorSignature(
    severity=ErrorSeverity.MAJOR,
    error_type=ErrorType.REMOVAL,
    context="containment_task",
    affected_categories=["spatial"],
    delta=0.65,
)

# Register fix when it works
registry.register(error, fix)

# Later: O(1) retrieval
same_pattern = ErrorSignature(
    severity=ErrorSeverity.MAJOR,
    error_type=ErrorType.REMOVAL,
    context="containment_task",
    affected_categories=["spatial"],
    delta=0.72,  # Different magnitude, same fix
)

retrieved = registry.lookup(same_pattern)  # ~1.5 μs
```

---

## Proactive Dreaming

Generate synthetic failures before deployment:

```python
from experience_memory import DreamGenerator

dreamer = DreamGenerator(registry=registry)
examples = dreamer.dream(
    concept=concept,
    strategies=["removal", "intensity", "substitution"],
    variants_per_strategy=3,
)
```

Pre-register fixes for edge cases that haven't occurred yet.

---

## Decay (Bounded Memory)

```python
from experience_memory import FixRegistry, DecayConfig

config = DecayConfig(
    exponential_rate=0.95,
    linear_rate=0.005,
    min_strength=0.05,
)

registry = FixRegistry(decay_config=config)

for epoch in range(100):
    pruned = registry.advance_epoch()
    # Fixes that keep getting used survive; others fade
```

---

## Benchmarks

```bash
python benchmarks/exp_lookup_latency.py --sizes 100,1000,10000,50000
python benchmarks/exp_error_elimination.py --length 1000 --unique 30
python benchmarks/exp_memory_decay.py --epochs 500 --rate 10
```

---

## Design

- Zero dependencies: Pure Python
- Memory bounded: Decay prevents unbounded growth
- Paradigm neutral: Works with any system that produces errors

---

MIT License
