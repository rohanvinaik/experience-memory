# Experience Memory

**Adapt from failures in O(1) time without gradient descent**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-green.svg)](#installation)

---

## The Problem

When deployed systems encounter errors, the standard remediation path is expensive:

| Approach | Time to Fix | Compute Cost | Risk |
|----------|-------------|--------------|------|
| Full retraining | Hours–Days | High (GPU clusters) | Catastrophic forgetting |
| Fine-tuning | Minutes–Hours | Medium | Distribution shift |
| Manual rules | Days (human time) | Low | Brittle, doesn't scale |

**What if the system could learn from each failure instantly, and apply that fix in O(1) time when similar errors recur?**

---

## Benchmark Results

All benchmarks run on Apple M-series CPU, pure Python, zero external dependencies.

### O(1) Lookup Latency (Verified)

| Registry Size | Mean Latency | P99 Latency | Ratio vs 100 |
|---------------|--------------|-------------|--------------|
| 100 | 1.33 µs | 1.50 µs | 1.00× |
| 1,000 | 1.37 µs | 4.46 µs | 1.03× |
| 10,000 | 1.41 µs | 1.79 µs | 1.06× |
| **50,000** | **1.63 µs** | **4.67 µs** | **1.22×** |

**Conclusion:** 500× more entries → only 1.22× slower. Effectively constant time.

### Repeated Error Elimination

| Metric | Value |
|--------|-------|
| Total errors processed | 1,000 |
| Final cache hit rate | **97.0%** |
| Steady-state window hit rate | **100%** |
| Learning phase (first 50 errors) | 80% hit rate |

After learning 30 unique patterns, the system achieves near-perfect cache hits on repeating errors.

### Memory Bounds Under Decay

| Configuration | Result |
|---------------|--------|
| Registration rate | 10 fixes/epoch |
| Decay rate | 0.97 exponential + 0.001 linear |
| Total registered over 500 epochs | 5,000 fixes |
| **Peak registry size** | **1,040** |
| Steady-state variance | 0.0 (perfectly stable) |

Despite 5,000 registrations, decay maintains bounded memory at ~1,040 entries.

### Hash Collision Analysis

| Metric | Value |
|--------|-------|
| Signatures tested | 50,000 |
| Unintended collision rate | **0.0000%** |
| Hash size | 16-char SHA256 (64 bits) |

Zero unintended collisions across 50,000 error signatures—fixes are applied to correct errors.

---

## This Approach

```
         ┌─────────────────────────────────────────────────┐
         │                  Experience Memory               │
         └─────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   ┌─────────┐          ┌─────────┐          ┌─────────┐
   │ Classify │          │  Dream  │          │  Decay  │
   │  Error   │          │ (Synth) │          │ (Forget)│
   └────┬────┘          └────┬────┘          └────┬────┘
        │                    │                    │
        ▼                    ▼                    ▼
   Error taxonomy      Generate synthetic    Prune unused
   → signature hash    edge cases before     fixes over time
   → O(1) lookup       deployment            (memory bounded)
```

1. **Classify** – Errors are typed (removal, substitution, intensity shift, etc.) and hashed into a 16-character signature
2. **Register** – When a fix works, store `signature → fix` in a hash table
3. **Lookup** – O(1) retrieval when the same error pattern recurs
4. **Dream** – Proactively generate synthetic corruptions to pre-register fixes *before* deployment
5. **Decay** – Unused fixes weaken over time and get pruned (bounded memory)

---

## Quick Start

### Installation

```bash
# From source
git clone https://github.com/rohanvinaik/experience-memory.git
cd experience-memory
pip install -e .
```

### Learn from a Failure

```python
from experience_memory import (
    FixRegistry, ErrorSignature, Fix,
    ErrorSeverity, ErrorType, FixType
)

registry = FixRegistry()

# An error occurred: boundary element was removed
error = ErrorSignature(
    severity=ErrorSeverity.MAJOR,
    error_type=ErrorType.REMOVAL,
    context="containment_task",
    affected_categories=["spatial"],
    delta=0.65,
)

# The fix that resolved it
fix = Fix(
    fix_type=FixType.ADD_ELEMENTS,
    elements_to_add={"boundary": 1.0},
    elements_to_remove=[],
    value_adjustments={},
    category_moves={},
    definition_supplement=None,
    decomposition=None,
)

# Register: O(1) insertion
registry.register(error, fix)

# Later: same error pattern recurs (even with different delta)
same_pattern = ErrorSignature(
    severity=ErrorSeverity.MAJOR,
    error_type=ErrorType.REMOVAL,
    context="containment_task",
    affected_categories=["spatial"],
    delta=0.72,  # Different magnitude, same fix applies
)

# O(1) retrieval (~1.5 µs)
retrieved = registry.lookup(same_pattern)
if retrieved:
    apply_fix(retrieved)
    registry.report_success(same_pattern)  # Reinforce
```

### Proactive Dreaming (Edge Case Generation)

```python
from experience_memory import DreamGenerator, FixRegistry

registry = FixRegistry()
dreamer = DreamGenerator(registry=registry)

# A concept the system has learned
concept = {
    "term": "containment",
    "activations": {"boundary": 1.0, "inside": 0.9, "wall": 0.7},
    "categories": ["spatial", "physical"],
}

# Generate synthetic failures before deployment
examples = dreamer.dream(
    concept=concept,
    strategies=["removal", "intensity", "substitution"],
    variants_per_strategy=3,
)

# Synthetic edge cases auto-registered as fixes
print(f"Pre-registered {registry.get_stats()['total_entries']} proactive fixes")
```

### Temporal Decay (Bounded Memory)

```python
from experience_memory import FixRegistry, DecayConfig

# Custom decay: faster forgetting
config = DecayConfig(
    exponential_rate=0.95,  # Per-epoch multiplier
    linear_rate=0.005,      # Per-epoch subtraction
    min_strength=0.05,      # Prune below 5%
)

registry = FixRegistry(decay_config=config)

# Each epoch, unused fixes decay
for epoch in range(100):
    pruned = registry.advance_epoch()
    # Fixes that keep getting used survive; others fade
```

---

## How It Works

### Error Signature Hashing

Errors are classified by type and context, then hashed for O(1) lookup:

```python
# Hash excludes delta (magnitude) so similar errors share fixes
def hash(self) -> str:
    key = f"{severity}:{error_type}:{context}:{sorted(categories)}"
    return sha256(key.encode()).hexdigest()[:16]
```

This means a "60% missing" error and "90% missing" error of the same type get the same fix—the magnitude is a symptom, not the cause.

### Decay Formula

Per epoch: `strength = strength × 0.97 − 0.001`

| Mechanism | Purpose |
|-----------|---------|
| Exponential (×0.97) | Geometric decay for smooth forgetting |
| Linear (−0.001) | Guarantees eventual pruning |
| Threshold (0.01) | Auto-remove when negligible |
| Success (+0.05) | Reinforce useful fixes |
| Failure (−0.10) | Rapidly prune bad fixes |

### Error Taxonomy

| Severity | Delta Range | Example |
|----------|-------------|---------|
| CATASTROPHIC | >90% | Complete concept loss |
| MAJOR | 50–90% | Significant distortion |
| MODERATE | 10–50% | Partial degradation |
| MINOR | <10% | Subtle shift |

| Error Type | Description |
|------------|-------------|
| REMOVAL | Elements missing |
| SUBSTITUTION | Wrong element used |
| INTENSITY_SHIFT | Value perturbation |
| STRUCTURAL_SWAP | Wrong category assignment |
| RELATIONSHIP_INVERSION | Opposite relationship |
| PARTIAL_DEFINITION | Incomplete specification |
| COMPONENT_REMOVAL | Missing sub-parts |

---

## Comparison with Standard Approaches

| Property | Retraining | Fine-tuning | RAG | **Experience Memory** |
|----------|------------|-------------|-----|----------------------|
| Learning method | Gradient descent | Gradient descent | Retrieval | Hash registration |
| Fix latency | Hours | Minutes | O(log n) | **O(1) ~1.5µs** |
| Memory overhead | Full model copy | Adapter weights | Vector DB | **~500B per fix** |
| Forgetting risk | High | Medium | None | **None** |
| Proactive defense | No | No | No | **Yes (dreaming)** |
| Gradient operations | Millions | Thousands | 0 | **0** |

---

## Run Benchmarks

```bash
# Install
pip install -e .

# O(1) verification
python benchmarks/exp_lookup_latency.py --sizes 100,1000,10000,50000

# Error elimination rate
python benchmarks/exp_error_elimination.py --length 1000 --unique 30

# Memory bounds
python benchmarks/exp_memory_decay.py --epochs 500 --rate 10

# Hash collision analysis
python benchmarks/exp_hash_collisions.py --num-errors 50000
```

---

## Architecture

```
experience_memory/
├── error_taxonomy.py    # ErrorSignature, severity classification
├── fix_registry.py      # O(1) hash-based registry
├── dream_generator.py   # Synthetic corruption generation
├── decay.py             # Temporal decay management
└── background_processor.py  # (Optional) Async daemon
```

**Design Principles:**
- **Zero dependencies:** Pure Python, no ML frameworks required
- **Paradigm neutral:** Works with any system that produces errors
- **Memory bounded:** Decay prevents unbounded growth
- **Proactive:** Dreaming generates edge cases before they occur in production

---

## Use Cases

1. **Production ML monitoring** – Cache fixes for recurring prediction errors
2. **Robotics control** – Learn sensor→actuator corrections without retraining
3. **Compiler optimization** – Cache peephole fixes for repeated code patterns
4. **Game AI** – Learn from player-discovered exploits
5. **Edge deployment** – Adapt on-device without connectivity to training infra

---

## Roadmap

- [ ] SQLite persistence for registry (crash recovery)
- [ ] Hierarchical error signatures (approximate matching)
- [ ] Streaming API for continuous integration
- [ ] Distributed registry for multi-agent systems
- [ ] Visualization dashboard for registry evolution

---

## Contributing

Contributions welcome! See the architecture above and pick an area:

- **Core:** Registry persistence, approximate matching
- **Dreaming:** Additional corruption strategies
- **Benchmarks:** Implement planned experiments
- **Integrations:** Wrap for specific use cases

---

## License

MIT License – see [LICENSE](LICENSE) for details.

---

## Citation

```bibtex
@software{experience_memory,
  author = {Vinaik, Rohan},
  title = {Experience Memory: O(1) Error-to-Fix Learning Without Gradient Descent},
  year = {2025},
  url = {https://github.com/rohanvinaik/experience-memory}
}
```
