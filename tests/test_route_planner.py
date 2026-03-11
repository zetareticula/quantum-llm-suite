"""Tests for route_planner — no external dependencies required."""

from __future__ import annotations

import pytest

from backend.route_planner import (
    DivideAndConquerRoutePlanner,
    GreedyRoutePlanner,
    RouteDecision,
    RouteOption,
    TaskRequest,
    default_score_weights,
    estimate_request_cost,
    estimate_token_count,
    explain_decision,
    get_route_planner,
    missing_capabilities,
    score_option,
    violates_constraints,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _opt(name: str, cost: float = 0.5, latency: int = 500, quality: float = 0.8,
         caps: set | None = None) -> RouteOption:
    return RouteOption(
        name=name,
        cost_per_1k_tokens=cost,
        est_latency_ms=latency,
        quality=quality,
        capabilities=caps or set(),
    )


def _req(prompt: str = "hello", caps: set | None = None, budget: float | None = None,
         latency: int | None = None, priority: int = 0) -> TaskRequest:
    return TaskRequest(
        prompt=prompt,
        required_capabilities=caps or set(),
        budget=budget,
        max_latency_ms=latency,
        priority=priority,
    )


# ---------------------------------------------------------------------------
# estimate_token_count
# ---------------------------------------------------------------------------

class TestEstimateTokenCount:
    def test_empty_returns_zero(self):
        assert estimate_token_count("") == 0

    def test_nonempty_at_least_one(self):
        assert estimate_token_count("hi") >= 1

    def test_longer_string_more_tokens(self):
        short = estimate_token_count("a" * 10)
        long_ = estimate_token_count("a" * 100)
        assert long_ > short


# ---------------------------------------------------------------------------
# estimate_request_cost
# ---------------------------------------------------------------------------

class TestEstimateRequestCost:
    def test_zero_cost_option(self):
        r = _req("hello")
        o = _opt("free", cost=0.0)
        assert estimate_request_cost(r, o) == 0.0

    def test_proportional_to_length(self):
        short = estimate_request_cost(_req("hi"), _opt("x", cost=1.0))
        long_ = estimate_request_cost(_req("hi " * 200), _opt("x", cost=1.0))
        assert long_ > short


# ---------------------------------------------------------------------------
# missing_capabilities / violates_constraints
# ---------------------------------------------------------------------------

class TestConstraints:
    def test_no_caps_required_no_missing(self):
        assert missing_capabilities(_req(), _opt("a")) == set()

    def test_missing_caps_detected(self):
        r = _req(caps={"quantum"})
        o = _opt("a", caps={"classical"})
        assert "quantum" in missing_capabilities(r, o)

    def test_caps_satisfied(self):
        r = _req(caps={"quantum"})
        o = _opt("a", caps={"quantum", "classical"})
        assert missing_capabilities(r, o) == set()

    def test_violates_latency(self):
        r = _req(latency=100)
        o = _opt("slow", latency=5000)
        bad, reasons = violates_constraints(r, o)
        assert bad
        assert "latency_exceeded" in reasons

    def test_violates_budget(self):
        r = _req(prompt="x" * 10000, budget=0.0001)
        o = _opt("expensive", cost=10.0)
        bad, reasons = violates_constraints(r, o)
        assert bad
        assert "budget_exceeded" in reasons

    def test_no_violation_when_within_constraints(self):
        r = _req(latency=9999, budget=999.0)
        o = _opt("ok", latency=100)
        bad, _ = violates_constraints(r, o)
        assert not bad


# ---------------------------------------------------------------------------
# score_option
# ---------------------------------------------------------------------------

class TestScoreOption:
    def test_higher_quality_wins(self):
        r = _req()
        high = _opt("hi-q", quality=0.95, cost=0.5, latency=500)
        low  = _opt("lo-q", quality=0.2,  cost=0.5, latency=500)
        s_hi, _ = score_option(r, high)
        s_lo, _ = score_option(r, low)
        assert s_hi > s_lo

    def test_lower_latency_wins_when_equal_quality(self):
        r = _req()
        fast = _opt("fast", quality=0.8, latency=100)
        slow = _opt("slow", quality=0.8, latency=5000)
        s_f, _ = score_option(r, fast)
        s_s, _ = score_option(r, slow)
        assert s_f > s_s

    def test_breakdown_keys_present(self):
        r = _req()
        o = _opt("x")
        _, breakdown = score_option(r, o)
        for key in ("quality", "est_latency_ms", "est_cost", "score", "weights"):
            assert key in breakdown


# ---------------------------------------------------------------------------
# default_score_weights
# ---------------------------------------------------------------------------

class TestDefaultScoreWeights:
    def test_zero_priority(self):
        w = default_score_weights(_req(priority=0))
        assert w["quality"] == 1.0
        assert w["cost"] == 1.0

    def test_high_priority_increases_latency_and_quality_weight(self):
        w_low  = default_score_weights(_req(priority=0))
        w_high = default_score_weights(_req(priority=9))
        assert w_high["quality"]  > w_low["quality"]
        assert w_high["latency"]  > w_low["latency"]


# ---------------------------------------------------------------------------
# GreedyRoutePlanner
# ---------------------------------------------------------------------------

class TestGreedyRoutePlanner:
    def test_raises_on_empty_options(self):
        planner = GreedyRoutePlanner()
        with pytest.raises(ValueError):
            planner.choose_route(_req(), [])

    def test_picks_feasible_option(self):
        planner = GreedyRoutePlanner()
        r = _req(caps={"quantum"})
        opts = [
            _opt("gpu",     caps={"classical"},        quality=0.9),
            _opt("quantum", caps={"quantum"},           quality=0.7),
        ]
        d = planner.choose_route(r, opts)
        assert d.option.name == "quantum"

    def test_picks_best_score_among_feasible(self):
        planner = GreedyRoutePlanner()
        r = _req()
        opts = [
            _opt("a", quality=0.6, latency=100),
            _opt("b", quality=0.95, latency=100),
            _opt("c", quality=0.3, latency=100),
        ]
        d = planner.choose_route(r, opts)
        assert d.option.name == "b"

    def test_allow_violations_falls_back_when_no_feasible(self):
        planner = GreedyRoutePlanner(allow_violations=True)
        r = _req(caps={"quantum"})
        opts = [_opt("gpu", caps={"classical"}, quality=0.9)]
        d = planner.choose_route(r, opts)
        assert d.option.name == "gpu"

    def test_raises_when_no_feasible_and_violations_not_allowed(self):
        planner = GreedyRoutePlanner(allow_violations=False)
        r = _req(caps={"quantum"})
        opts = [_opt("gpu", caps={"classical"})]
        with pytest.raises(ValueError):
            planner.choose_route(r, opts)

    def test_plan_routes_returns_one_per_request(self):
        planner = GreedyRoutePlanner()
        requests = [_req("a"), _req("b"), _req("c")]
        opts = [_opt("x", quality=0.8)]
        decisions = planner.plan_routes(requests, opts)
        assert len(decisions) == 3
        assert all(isinstance(d, RouteDecision) for d in decisions)


# ---------------------------------------------------------------------------
# DivideAndConquerRoutePlanner
# ---------------------------------------------------------------------------

class TestDivideAndConquerRoutePlanner:
    def _options(self):
        return [_opt("x", quality=0.8, caps={"quantum"}), _opt("y", quality=0.6)]

    def test_empty_requests_returns_empty(self):
        planner = DivideAndConquerRoutePlanner()
        assert planner.plan_routes([], self._options()) == []

    def test_single_request_returns_one_decision(self):
        planner = DivideAndConquerRoutePlanner()
        result = planner.plan_routes([_req()], self._options())
        assert len(result) == 1

    def test_large_batch_returns_correct_count(self):
        planner = DivideAndConquerRoutePlanner(max_group_size=3)
        requests = [_req(f"prompt {i}") for i in range(20)]
        opts = [_opt("x")]
        decisions = planner.plan_routes(requests, opts)
        assert len(decisions) == 20

    def test_groups_by_capability_signature(self):
        """Requests with different capability requirements should both be routed."""
        planner = DivideAndConquerRoutePlanner(max_group_size=2)
        requests = [
            _req("a", caps={"quantum"}),
            _req("b", caps={"quantum"}),
            _req("c", caps=set()),
            _req("d", caps=set()),
        ]
        opts = [_opt("q", caps={"quantum"}, quality=0.9), _opt("any")]
        decisions = planner.plan_routes(requests, opts)
        assert len(decisions) == 4

    def test_choose_route_delegates_to_leaf(self):
        planner = DivideAndConquerRoutePlanner()
        d = planner.choose_route(_req(), [_opt("x")])
        assert isinstance(d, RouteDecision)


# ---------------------------------------------------------------------------
# explain_decision
# ---------------------------------------------------------------------------

class TestExplainDecision:
    def test_output_has_required_keys(self):
        planner = GreedyRoutePlanner()
        r = _req(caps={"quantum"})
        opts = [_opt("q", caps={"quantum"}, quality=0.9)]
        d = planner.choose_route(r, opts)
        exp = explain_decision(r, d)
        for key in ("selected", "score", "violates_constraints", "violations", "reasons"):
            assert key in exp

    def test_no_violation_for_satisfied_request(self):
        planner = GreedyRoutePlanner()
        r = _req(caps={"quantum"})
        opts = [_opt("q", caps={"quantum"})]
        d = planner.choose_route(r, opts)
        exp = explain_decision(r, d)
        assert exp["violates_constraints"] is False


# ---------------------------------------------------------------------------
# get_route_planner factory
# ---------------------------------------------------------------------------

class TestGetRoutePlanner:
    def test_greedy_alias(self):
        assert isinstance(get_route_planner("greedy"), GreedyRoutePlanner)
        assert isinstance(get_route_planner("fast"), GreedyRoutePlanner)

    def test_dac_aliases(self):
        for name in ("divide_and_conquer", "dac", "hierarchical"):
            assert isinstance(get_route_planner(name), DivideAndConquerRoutePlanner)

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            get_route_planner("unknown_strategy")
