from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
from loguru import logger


@dataclass
class EvalLogger:
    context: str = "eval"

    def log(self, metrics: Dict[str, Any]) -> None:
        logger.info(f"[{self.context}] {metrics}")
