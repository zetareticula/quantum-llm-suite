
from __future__ import annotations

import logging
import os
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from apscheduler.schedulers.background import BackgroundScheduler


logger = logging.getLogger(__name__)


@dataclass
class AnnealingState:
    """Simple annealing state for route weights.

    This is a lightweight stand-in for more advanced online optimization.
    """

    quality_weight: float = 1.0
    latency_weight: float = 1.0
    cost_weight: float = 1.0
    updated_at: float = 0.0


_SCHEDULER: Optional[BackgroundScheduler] = None
_STATE = AnnealingState(updated_at=time.time())
_LOCK = threading.Lock()


def get_annealing_state() -> AnnealingState:
    with _LOCK:
        return AnnealingState(**_STATE.__dict__)


def _anneal_step() -> None:
    """Periodic small random walk over weights.

    Numerical method:
    - Multiplicative jitter with clipping, to emulate annealing over objective weights.
    """

    with _LOCK:
        jitter = float(os.getenv("ANNEAL_JITTER", "0.05"))
        for attr in ("quality_weight", "latency_weight", "cost_weight"):
            v = getattr(_STATE, attr)
            v *= 1.0 + random.uniform(-jitter, jitter)
            v = max(0.1, min(10.0, v))
            setattr(_STATE, attr, v)
        _STATE.updated_at = time.time()
    logger.info(
        "Annealed weights: quality=%.3f latency=%.3f cost=%.3f",
        _STATE.quality_weight,
        _STATE.latency_weight,
        _STATE.cost_weight,
    )


def start_scheduler() -> BackgroundScheduler:
    global _SCHEDULER
    if _SCHEDULER is not None:
        return _SCHEDULER

    interval_minutes = int(os.getenv("ANNEAL_INTERVAL_MINUTES", "30"))
    sched = BackgroundScheduler(daemon=True)
    sched.add_job(_anneal_step, "interval", minutes=max(1, interval_minutes), id="anneal_step", replace_existing=True)
    sched.start()
    _SCHEDULER = sched
    return sched


def stop_scheduler() -> None:
    global _SCHEDULER
    if _SCHEDULER is None:
        return
    try:
        _SCHEDULER.shutdown(wait=False)
    finally:
        _SCHEDULER = None