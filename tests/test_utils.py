"""Tests for backend/utils.py — no model downloads or quantum hardware required."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# validate_api_key  (single-arg: key string only)
# ---------------------------------------------------------------------------

class TestValidateApiKey:
    def test_valid_key_in_set_returns_true(self):
        from backend.utils import validate_api_key
        with patch("backend.utils.API_KEYS", {"my-token"}):
            assert validate_api_key("my-token") is True

    def test_wrong_key_returns_false(self):
        from backend.utils import validate_api_key
        with patch("backend.utils.API_KEYS", {"real-token"}):
            assert validate_api_key("bad-token") is False

    def test_empty_string_always_false(self):
        from backend.utils import validate_api_key
        with patch("backend.utils.API_KEYS", {"real-token"}):
            assert validate_api_key("") is False

    def test_env_api_key_overrides_set(self):
        from backend.utils import validate_api_key
        with (
            patch("backend.utils.API_KEYS", {"set-token"}),
            patch("os.getenv", side_effect=lambda k, *a: "env-token" if k == "API_KEY" else None),
        ):
            assert validate_api_key("env-token") is True


# ---------------------------------------------------------------------------
# compute_benchmark  (classical_output, quantum_output, classical_seconds, quantum_seconds) -> Benchmark
# ---------------------------------------------------------------------------

class TestComputeBenchmark:
    def test_speedup_calculation(self):
        from backend.utils import compute_benchmark
        bm = compute_benchmark("same", "same", classical_seconds=1.2, quantum_seconds=0.4)
        assert bm.speedup == pytest.approx(1.2 / 0.4)

    def test_zero_quantum_clamped_no_crash(self):
        from backend.utils import compute_benchmark
        bm = compute_benchmark("a", "b", classical_seconds=1.0, quantum_seconds=0.0)
        assert bm.speedup > 0

    def test_divergence_zero_for_identical_outputs(self):
        from backend.utils import compute_benchmark
        bm = compute_benchmark("hello", "hello", 0.5, 0.5)
        assert bm.divergence == pytest.approx(0.0)

    def test_divergence_positive_for_different_outputs(self):
        from backend.utils import compute_benchmark
        bm = compute_benchmark("short", "a much longer output text", 0.5, 0.3)
        assert bm.divergence > 0.0

    def test_dataclass_fields_present(self):
        from backend.utils import Benchmark, compute_benchmark
        bm = compute_benchmark("x", "y", 0.1, 0.2)
        assert hasattr(bm, "classical_seconds")
        assert hasattr(bm, "quantum_seconds")
        assert hasattr(bm, "speedup")
        assert hasattr(bm, "divergence")


# ---------------------------------------------------------------------------
# HF_MODELS / QUANTUM_BACKENDS constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_hf_models_nonempty_list(self):
        from backend.utils import HF_MODELS
        assert isinstance(HF_MODELS, list)
        assert len(HF_MODELS) >= 1

    def test_quantum_backends_nonempty_list(self):
        from backend.utils import QUANTUM_BACKENDS
        assert isinstance(QUANTUM_BACKENDS, list)
        assert len(QUANTUM_BACKENDS) >= 1

    def test_local_in_quantum_backends(self):
        from backend.utils import QUANTUM_BACKENDS
        assert "Local" in QUANTUM_BACKENDS


# ---------------------------------------------------------------------------
# get_model / get_tokenizer caching  (lazy-load via _lazy_import)
# ---------------------------------------------------------------------------

class TestModelCache:
    def test_get_model_called_once_cached(self):
        """Calling get_model twice with the same args should only load once."""
        from backend import utils as _utils
        _utils._MODEL_CACHE.clear()

        mock_model = MagicMock(name="model")
        mock_model.eval = MagicMock()
        transformers_mock = MagicMock()
        transformers_mock.AutoModelForCausalLM.from_pretrained.return_value = mock_model

        with patch("backend.utils._lazy_import", return_value=transformers_mock):
            m1 = _utils.get_model("THUDM/glm-4-9b-chat", "fp16")
            m2 = _utils.get_model("THUDM/glm-4-9b-chat", "fp16")

        assert m1 is m2
        transformers_mock.AutoModelForCausalLM.from_pretrained.assert_called_once()

    def test_get_tokenizer_called_once_cached(self):
        """Calling get_tokenizer twice with the same name should only load once."""
        from backend import utils as _utils
        _utils._TOKENIZER_CACHE.clear()

        mock_tok = MagicMock(name="tokenizer")
        mock_tok.pad_token = None
        mock_tok.eos_token = "</s>"
        transformers_mock = MagicMock()
        transformers_mock.AutoTokenizer.from_pretrained.return_value = mock_tok

        with patch("backend.utils._lazy_import", return_value=transformers_mock):
            t1 = _utils.get_tokenizer("THUDM/glm-4-9b-chat")
            t2 = _utils.get_tokenizer("THUDM/glm-4-9b-chat")

        assert t1 is t2
        transformers_mock.AutoTokenizer.from_pretrained.assert_called_once()

    def test_different_quant_levels_cached_separately(self):
        """fp16 and int8 for the same model should be separate cache entries."""
        from backend import utils as _utils
        _utils._MODEL_CACHE.clear()

        mock_fp16 = MagicMock(name="model_fp16")
        mock_fp16.eval = MagicMock()
        mock_int8 = MagicMock(name="model_int8")
        mock_int8.eval = MagicMock()
        call_count = 0

        def _from_pretrained(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return mock_fp16 if call_count == 1 else mock_int8

        transformers_mock = MagicMock()
        transformers_mock.AutoModelForCausalLM.from_pretrained.side_effect = _from_pretrained

        with patch("backend.utils._lazy_import", return_value=transformers_mock):
            m_fp16 = _utils.get_model("THUDM/glm-4-9b-chat", "fp16")
            m_int8 = _utils.get_model("THUDM/glm-4-9b-chat", "int8")

        assert m_fp16 is not m_int8
