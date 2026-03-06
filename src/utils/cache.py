"""
LLM Result Cache — file-based, keyed by SHA256(prompt + model).
Avoids redundant LLM calls across repeated pipeline runs.
"""
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class LLMCache:
    """
    Simple file-based cache for LLM responses.
    Key = SHA256(model + prompt). Value = JSON response.
    """

    def __init__(self, cache_dir: str = ".refinery/cache", enabled: bool = True):
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, model: str, prompt: str) -> str:
        return hashlib.sha256(f"{model}::{prompt}".encode("utf-8")).hexdigest()

    def get(self, model: str, prompt: str) -> Optional[str]:
        """Return cached response string, or None if not cached."""
        if not self.enabled:
            return None
        path = self.cache_dir / f"{self._key(model, prompt)}.json"
        if path.exists():
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                logger.debug(f"Cache HIT for {path.name[:12]}...")
                return data.get("response")
            except Exception:
                return None
        return None

    def put(self, model: str, prompt: str, response: str) -> None:
        """Store a response in the cache."""
        if not self.enabled:
            return
        path = self.cache_dir / f"{self._key(model, prompt)}.json"
        try:
            with open(path, "w") as f:
                json.dump({"model": model, "response": response}, f)
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")
