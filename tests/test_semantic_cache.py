"""Tests for semantic cache."""

import pytest
import tempfile
from experience_memory.semantic_cache import SemanticCache

class TestSemanticCache:
    
    @pytest.fixture
    def cache(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            return SemanticCache(f.name)
            
    def test_cache_miss_returns_none(self, cache):
        assert cache.get("unknown") is None
        
    def test_cache_hit_returns_data(self, cache):
        text = "Hello World"
        data = {"foo": "bar", "val": 123}
        
        cache.set(text, data)
        result = cache.get(text)
        
        assert result == data
