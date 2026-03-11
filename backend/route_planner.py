
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class RouteOption:
    name: str
    cost_per_1k_tokens: float = 0.0
    est_latency_ms: int = 0
    quality: float = 0.0
    capabilities: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskRequest:
    prompt: str
    required_capabilities: Set[str] = field(default_factory=set)
    budget: Optional[float] = None
    max_latency_ms: Optional[int] = None
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RouteDecision:
    option: RouteOption
    score: float
    reasons: Dict[str, Any] = field(default_factory=dict)


def estimate_token_count(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_request_cost(request: TaskRequest, option: RouteOption) -> float:
    tokens = estimate_token_count(request.prompt)
    return (tokens / 1000.0) * float(option.cost_per_1k_tokens)


def missing_capabilities(request: TaskRequest, option: RouteOption) -> Set[str]:
    return set(request.required_capabilities) - set(option.capabilities)


def violates_constraints(request: TaskRequest, option: RouteOption) -> Tuple[bool, Dict[str, Any]]:
    reasons: Dict[str, Any] = {}
    missing = missing_capabilities(request, option)
    if missing:
        reasons["missing_capabilities"] = sorted(missing)
    if request.max_latency_ms is not None and option.est_latency_ms > request.max_latency_ms:
        reasons["latency_exceeded"] = {
            "max_latency_ms": request.max_latency_ms,
            "est_latency_ms": option.est_latency_ms,
        }
    if request.budget is not None:
        est_cost = estimate_request_cost(request, option)
        if est_cost > request.budget:
            reasons["budget_exceeded"] = {
                "budget": request.budget,
                "est_cost": est_cost,
            }
    return (len(reasons) > 0), reasons


def default_score_weights(request: TaskRequest) -> Dict[str, float]:
    priority = max(0, int(request.priority))
    latency_w = 1.0 + min(3.0, priority / 3.0)
    quality_w = 1.0 + min(4.0, priority / 2.0)
    cost_w = 1.0
    return {"quality": quality_w, "latency": latency_w, "cost": cost_w}


def score_option(
    request: TaskRequest,
    option: RouteOption,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, Any]]:
    w = weights or default_score_weights(request)
    est_cost = estimate_request_cost(request, option)
    latency_penalty = float(option.est_latency_ms) / 1000.0
    cost_penalty = est_cost
    quality_value = float(option.quality)
    score = (w.get("quality", 1.0) * quality_value) - (w.get("latency", 1.0) * latency_penalty) - (
        w.get("cost", 1.0) * cost_penalty
    )
    breakdown = {
        "weights": dict(w),
        "quality": quality_value,
        "est_latency_ms": int(option.est_latency_ms),
        "latency_penalty": latency_penalty,
        "est_cost": est_cost,
        "cost_penalty": cost_penalty,
        "score": score,
    }
    return score, breakdown


def explain_decision(request: TaskRequest, decision: RouteDecision) -> Dict[str, Any]:
    violates, violations = violates_constraints(request, decision.option)
    return {
        "selected": decision.option.name,
        "score": decision.score,
        "violates_constraints": violates,
        "violations": violations,
        "reasons": dict(decision.reasons),
    }


class BaseRoutePlanner:
    def choose_route(self, request: TaskRequest, options: Sequence[RouteOption]) -> RouteDecision:
        raise NotImplementedError()

    def plan_routes(self, requests: Sequence[TaskRequest], options: Sequence[RouteOption]) -> List[RouteDecision]:
        return [self.choose_route(r, options) for r in requests]


class GreedyRoutePlanner(BaseRoutePlanner):
    def __init__(
        self,
        weights_fn: Optional[Callable[[TaskRequest], Dict[str, float]]] = None,
        allow_violations: bool = False,
    ) -> None:
        self._weights_fn = weights_fn or default_score_weights
        self._allow_violations = bool(allow_violations)

    def choose_route(self, request: TaskRequest, options: Sequence[RouteOption]) -> RouteDecision:
        if not options:
            raise ValueError("No route options provided")

        best: Optional[RouteDecision] = None
        best_any: Optional[RouteDecision] = None

        for opt in options:
            score, breakdown = score_option(request, opt, self._weights_fn(request))
            violates, violations = violates_constraints(request, opt)
            reasons = {"score_breakdown": breakdown, "constraint_violations": violations}
            decision = RouteDecision(option=opt, score=score, reasons=reasons)

            if best_any is None or decision.score > best_any.score:
                best_any = decision
            if not violates:
                if best is None or decision.score > best.score:
                    best = decision

        if best is not None:
            return best
        if self._allow_violations and best_any is not None:
            return best_any
        raise ValueError("No feasible route option satisfies constraints")


def _capability_signature(req: TaskRequest) -> Tuple[str, ...]:
    return tuple(sorted(req.required_capabilities))


class DivideAndConquerRoutePlanner(BaseRoutePlanner):
    def __init__(
        self,
        leaf_planner: Optional[BaseRoutePlanner] = None,
        max_group_size: int = 8,
    ) -> None:
        self._leaf = leaf_planner or GreedyRoutePlanner()
        self._max_group_size = max(1, int(max_group_size))

    def plan_routes(self, requests: Sequence[TaskRequest], options: Sequence[RouteOption]) -> List[RouteDecision]:
        if not requests:
            return []
        if len(requests) <= self._max_group_size:
            return self._leaf.plan_routes(requests, options)

        grouped: Dict[Tuple[str, ...], List[TaskRequest]] = {}
        for r in requests:
            grouped.setdefault(_capability_signature(r), []).append(r)

        result: List[RouteDecision] = []
        for _, group in sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            result.extend(self._plan_group(group, options))
        return result

    def _plan_group(self, group: Sequence[TaskRequest], options: Sequence[RouteOption]) -> List[RouteDecision]:
        if len(group) <= self._max_group_size:
            return self._leaf.plan_routes(group, options)
        mid = len(group) // 2
        left = self._plan_group(group[:mid], options)
        right = self._plan_group(group[mid:], options)
        return left + right

    def choose_route(self, request: TaskRequest, options: Sequence[RouteOption]) -> RouteDecision:
        return self._leaf.choose_route(request, options)


def get_route_planner(name: str = "greedy", **kwargs: Any) -> BaseRoutePlanner:
    key = (name or "").strip().lower()
    if key in {"greedy", "fast"}:
        return GreedyRoutePlanner(**kwargs)
    if key in {"divide_and_conquer", "dac", "hierarchical"}:
        return DivideAndConquerRoutePlanner(**kwargs)
    raise ValueError(f"Unknown route planner: {name}")