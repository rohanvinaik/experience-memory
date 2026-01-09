"""
Semantic cache implementation using experience-memory.
"""

import hashlib
import json
from typing import Optional, Dict

from experience_memory.registry import FixRegistry
from experience_memory.types import ErrorSignature, Fix, ErrorType, FixType

class SemanticCache:
    """O(1) cache for semantic analysis results."""
    
    def __init__(self, db_path: str = "/tmp/semantic_cache.db"):
        self.registry = FixRegistry(db_path)
        
    def _get_sig(self, text: str) -> ErrorSignature:
        h = hashlib.sha256(text.encode()).hexdigest()
        return ErrorSignature(
            severity=1,
            error_type=ErrorType.MISSING_CONTEXT,
            context=f"sem:{h}",
            affected_categories=["semantic_cache"],
            delta=0.0
        )
        
    def get(self, text: str) -> Optional[Dict]:
        """Retrieve cached analysis."""
        sig = self._get_sig(text)
        fix = self.registry.lookup(sig)
        if fix:
            return json.loads(fix.definition_supplement)
        return None
        
    def set(self, text: str, analysis: Dict):
        """Cache analysis result."""
        sig = self._get_sig(text)
        fix = Fix(
            fix_type=FixType.COMPLETE_DEFINITION,
            definition_supplement=json.dumps(analysis)
        )
        self.registry.register(sig, fix)

def main():
    # Simple demo
    cache = SemanticCache()
    text = "The quick brown fox."
    analysis = {"entropy": 1.2, "dimension": "ACTION"}
    
    print(f"Caching: {text} -> {analysis}")
    cache.set(text, analysis)
    
    retrieved = cache.get(text)
    print(f"Retrieved: {retrieved}")
    
    if retrieved == analysis:
        print("Success!")
    else:
        print("Failure.")

if __name__ == "__main__":
    main()
