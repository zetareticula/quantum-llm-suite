"""
Tests for quantum_dispatcher and gpu_dispatcher.

Patching strategy:
- `backend.quantum_dispatcher.get_tokenizer` / `get_model` — because the module
  binds these names at import time via `from .utils import ...`.
- Same for `backend.gpu_dispatcher.*`.
- `torch.randn` and `torch.Generator` are patched only for the quantum_embedding
  path to avoid dtype/device issues with MagicMock tensors.
- All HF model downloads and quantum SDK calls are mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Minimal mock factories
# ---------------------------------------------------------------------------

def _tensor(shape: list | None = None, device: str = "cpu"):
    t = MagicMock(name="tensor")
    t.device = device
    t.shape = shape or [1, 4, 512]
    t.to = MagicMock(return_value=t)
    t.__getitem__ = MagicMock(return_value=t)
    t.__matmul__ = MagicMock(return_value=t)
    t.__add__ = MagicMock(return_value=t)
    t.__mul__ = MagicMock(return_value=t)
    t.__rmul__ = MagicMock(return_value=t)
    t.transpose = MagicMock(return_value=t)
    t.dtype = MagicMock()
    return t


def _tokenizer(decoded: str = "Generated text"):
    tok = MagicMock(name="tokenizer")
    tok.pad_token = "<pad>"
    tok.eos_token = "</s>"
    ids = _tensor()
    attn = _tensor()
    tok.return_value = {"input_ids": ids, "attention_mask": attn}
    tok.decode = MagicMock(return_value=decoded)
    return tok


def _model():
    m = MagicMock(name="model")
    m.device = "cpu"
    out = _tensor()
    m.generate = MagicMock(return_value=out)
    emb = _tensor(shape=[1, 4, 512])
    embedder = MagicMock(return_value=emb)
    m.get_input_embeddings = MagicMock(return_value=embedder)
    m.config = MagicMock()
    m.config.hidden_size = 512
    return m


# ---------------------------------------------------------------------------
# _seed_from_counts
# ---------------------------------------------------------------------------

class TestSeedFromCounts:
    def test_empty_gives_zero(self):
        from backend.quantum_dispatcher import _seed_from_counts
        assert _seed_from_counts({}) == 0

    def test_bitstring_base2(self):
        from backend.quantum_dispatcher import _seed_from_counts
        assert _seed_from_counts({"0101": 1}) == 5

    def test_all_zeros(self):
        from backend.quantum_dispatcher import _seed_from_counts
        assert _seed_from_counts({"0000": 1}) == 0

    def test_non_bitstring_hashed(self):
        from backend.quantum_dispatcher import _seed_from_counts
        result = _seed_from_counts({3: 1})
        assert isinstance(result, int) and result >= 0

    def test_picks_max_count(self):
        from backend.quantum_dispatcher import _seed_from_counts
        # "01" count=5 wins over "11" count=1 → seed = int("01",2) = 1
        assert _seed_from_counts({"11": 1, "01": 5}) == 1


# ---------------------------------------------------------------------------
# zr_quantum_compiler
# ---------------------------------------------------------------------------

class TestZrQuantumCompiler:
    def test_invalid_provider_raises(self):
        from backend.quantum_dispatcher import zr_quantum_compiler
        with pytest.raises(ValueError, match="Invalid provider"):
            zr_quantum_compiler("OPENQASM 3.0;", "BadProvider")

    def test_fallback_on_exception(self):
        from backend.quantum_dispatcher import zr_quantum_compiler
        with patch("backend.quantum_dispatcher._get_qiskit_backend", side_effect=RuntimeError("sim broken")):
            result = zr_quantum_compiler("OPENQASM 3.0;", "Local")
        assert result == {"0": 1}

    def test_local_provider_returns_dict(self):
        from backend.quantum_dispatcher import zr_quantum_compiler

        mock_counts = {"0000": 1}
        mock_run_result = MagicMock()
        mock_run_result.get_counts.return_value = mock_counts
        mock_job = MagicMock()
        mock_job.result.return_value = mock_run_result
        mock_backend = MagicMock()
        mock_backend.run.return_value = mock_job

        with (
            patch("backend.quantum_dispatcher._get_qiskit_backend", return_value=mock_backend),
            patch("qiskit.transpile", return_value=MagicMock()),
            patch("qiskit.qasm3.loads", return_value=MagicMock()),
        ):
            result = zr_quantum_compiler("OPENQASM 3.0;", "Local")

        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# quantum_inference
# ---------------------------------------------------------------------------

class TestQuantumInference:
    _PATCHES = {
        "build": "backend.quantum_dispatcher._build_seed_circuit_qasm3",
        "compile": "backend.quantum_dispatcher.zr_quantum_compiler",
        "tok": "backend.quantum_dispatcher.get_tokenizer",
        "model": "backend.quantum_dispatcher.get_model",
    }

    def _run(self, mode: str, quant_level: str = "fp16", provider: str = "Local",
             model_name: str = "THUDM/glm-4-9b-chat"):
        tok = _tokenizer()
        mod = _model()
        proj_t = _tensor(shape=[512, 256])
        with (
            patch(self._PATCHES["build"], return_value="OPENQASM 3.0;"),
            patch(self._PATCHES["compile"], return_value={"0101": 1}),
            patch(self._PATCHES["tok"], return_value=tok),
            patch(self._PATCHES["model"], return_value=mod),
            patch("torch.Generator", return_value=MagicMock(manual_seed=MagicMock())),
            patch("torch.randn", return_value=proj_t),
        ):
            from backend.quantum_dispatcher import quantum_inference
            return quantum_inference("Test prompt", provider, False, model_name, quant_level, mode)

    def test_classical_output_prefix(self):
        out, elapsed = self._run("classical")
        assert out.startswith("Quantum (Local, fp16, classical) output:")
        assert elapsed >= 0.0

    def test_classical_contains_mod_tag(self):
        out, _ = self._run("classical")
        assert "(Classical mod:" in out

    def test_distillation_prefix(self):
        out, elapsed = self._run("quantum_distillation")
        assert out.startswith("Quantum (Local, fp16, quantum_distillation) output:")
        assert elapsed >= 0.0

    def test_distillation_contains_tag(self):
        out, _ = self._run("quantum_distillation")
        assert "Quantum distillation mod:" in out

    def test_embedding_prefix(self):
        out, elapsed = self._run("quantum_embedding")
        assert out.startswith("Quantum (Local, fp16, quantum_embedding) output:")
        assert elapsed >= 0.0

    def test_embedding_contains_tag(self):
        out, _ = self._run("quantum_embedding")
        assert "Quantum embedding compression:" in out

    def test_invalid_model_uses_default(self):
        out, _ = self._run("classical", model_name="NonExistentModel/x")
        # Should still produce a valid Quantum output string (model falls back to HF_MODELS[0])
        assert "Quantum" in out

    def test_error_caught_and_returned(self):
        with (
            patch("backend.quantum_dispatcher._build_seed_circuit_qasm3", side_effect=RuntimeError("boom")),
        ):
            from backend.quantum_dispatcher import quantum_inference
            out, elapsed = quantum_inference("Hi", "Local", False, "THUDM/glm-4-9b-chat", "fp16", "classical")
        assert "Error during quantum inference:" in out
        assert elapsed >= 0.0


# ---------------------------------------------------------------------------
# gpu_inference
# ---------------------------------------------------------------------------

class TestGpuInference:
    def _run(self, mode: str, quant_level: str = "fp16", cloud: str = "AWS",
             model_name: str = "THUDM/glm-4-9b-chat"):
        tok = _tokenizer("GPU text")
        mod = _model()
        proj_t = _tensor(shape=[512, 256])
        with (
            patch("backend.gpu_dispatcher.get_tokenizer", return_value=tok),
            patch("backend.gpu_dispatcher.get_model", return_value=mod),
            patch("torch.Generator", return_value=MagicMock(manual_seed=MagicMock())),
            patch("torch.randn", return_value=proj_t),
        ):
            from backend.gpu_dispatcher import gpu_inference
            return gpu_inference("Test prompt", cloud, model_name, quant_level, mode)

    def test_classical_prefix(self):
        out, elapsed = self._run("classical")
        assert out.startswith("Classical (AWS, fp16, classical) output:")
        assert elapsed >= 0.0

    def test_distillation_suffix(self):
        out, elapsed = self._run("quantum_distillation")
        assert out.startswith("Classical (AWS, fp16, quantum_distillation) output:")
        assert "(Classical distillation sim)" in out
        assert elapsed >= 0.0

    def test_embedding_prefix(self):
        out, elapsed = self._run("quantum_embedding")
        assert out.startswith("Classical (AWS, fp16, quantum_embedding) output:")
        assert "(Embedding sim: rank=" in out
        assert elapsed >= 0.0

    def test_error_handling(self):
        with patch("backend.gpu_dispatcher.get_tokenizer", side_effect=OSError("no model")):
            from backend.gpu_dispatcher import gpu_inference
            out, elapsed = gpu_inference("Hi", "GCP", "THUDM/glm-4-9b-chat", "fp16", "classical")
        assert "Error:" in out
        assert out.startswith("Classical (GCP, fp16, classical) output:")
        assert elapsed >= 0.0

    def test_invalid_model_uses_default(self):
        out, _ = self._run("classical", model_name="NonExistentModel/x")
        assert "Classical" in out
