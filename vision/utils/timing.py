import time
from typing import Optional


class ComponentTimer:
    """Context manager that measures elapsed milliseconds for a code block."""

    def __init__(self, name: str = ""):
        self.name = name
        self.elapsed_ms: float = 0.0
        self._start: Optional[float] = None

    def __enter__(self) -> "ComponentTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
