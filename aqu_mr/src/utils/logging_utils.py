"""
Logging and Debug Utilities for AQU-MR
Provides structured logging, console formatting, and debug event recording.
"""

import sys
import logging
from typing import Any, Dict, Optional


def setup_logger(name: str = "AQU-MR", level: int = logging.INFO) -> logging.Logger:
    """Configures a clean, formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


class DebugTracer:
    """Records full execution lifecycle for inspection and web UI debug drawer."""
    def __init__(self):
        self.events = []
        self.debug_data: Dict[str, Any] = {}

    def log_event(self, stage: str, data: Any):
        self.events.append({"stage": stage, "data": data})
        self.debug_data[stage] = data

    def get_trace(self) -> Dict[str, Any]:
        return {
            "events": self.events,
            "data": self.debug_data
        }

    def print_summary(self):
        print("\n" + "=" * 60)
        print("[DEBUG] AQU-MR Execution Lifecycle Summary")
        print("=" * 60)
        for event in self.events:
            print(f"[{event['stage']}]: {str(event['data'])[:120]}...")
        print("=" * 60 + "\n")
