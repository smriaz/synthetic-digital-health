import numpy as np

class RNG:
    """Reproducible RNG wrapper that supports deterministic streams per module."""
    def __init__(self, seed: int):
        self._root = np.random.default_rng(seed)

    def child(self, tag: str) -> np.random.Generator:
        # deterministic child seed from tag
        h = abs(hash(tag)) % (2**32)
        child_seed = int(self._root.integers(0, 2**32-1) ^ h)
        return np.random.default_rng(child_seed)
