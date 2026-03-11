"""Root conftest.py — ensures the repo root is on sys.path so `backend` is importable,
and pre-populates sys.modules with lightweight stubs for heavy optional deps
(torch, qiskit, transformers, pyquil, cirq, braket) so that:
  1. `import torch` inside production code returns the mock (no install needed).
  2. `patch("torch.inference_mode", ...)` resolves without ImportError.
Tests that need specific behaviour override these stubs via additional patch() calls.
"""

import sys
import os
import types
from contextlib import contextmanager
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(__file__))


def _ctx_mock():
    """A MagicMock that also works as a `with` context manager."""
    m = MagicMock()
    m.__enter__ = MagicMock(return_value=None)
    m.__exit__ = MagicMock(return_value=False)
    return m


def _make_tensor_mock():
    t = MagicMock(name="tensor")
    t.device = "cpu"
    t.shape = [1, 4, 512]
    t.dtype = MagicMock()
    t.to = MagicMock(return_value=t)
    t.__getitem__ = MagicMock(return_value=t)
    t.__matmul__ = MagicMock(return_value=t)
    t.__add__ = MagicMock(return_value=t)
    t.__mul__ = MagicMock(return_value=t)
    t.__rmul__ = MagicMock(return_value=t)
    t.transpose = MagicMock(return_value=t)
    return t


# ---------------------------------------------------------------------------
# torch stub
# ---------------------------------------------------------------------------
_torch = types.ModuleType("torch")
_torch.inference_mode = MagicMock(return_value=_ctx_mock())

_gen = MagicMock()
_gen.manual_seed = MagicMock(return_value=_gen)
_torch.Generator = MagicMock(return_value=_gen)
_torch.randn = MagicMock(return_value=_make_tensor_mock())
_torch.no_grad = MagicMock(return_value=_ctx_mock())

sys.modules.setdefault("torch", _torch)


# ---------------------------------------------------------------------------
# qiskit stubs
# ---------------------------------------------------------------------------
for _mod in (
    "qiskit",
    "qiskit.qasm3",
    "qiskit_aer",
    "qiskit_ibm_runtime",
):
    sys.modules.setdefault(_mod, types.ModuleType(_mod))

# Ensure qiskit.qasm3.loads / dumps attributes exist
_qiskit = sys.modules["qiskit"]
_qiskit_qasm3 = sys.modules["qiskit.qasm3"]
if not hasattr(_qiskit_qasm3, "loads"):
    _qiskit_qasm3.loads = MagicMock(return_value=MagicMock())
if not hasattr(_qiskit_qasm3, "dumps"):
    _qiskit_qasm3.dumps = MagicMock(return_value="OPENQASM 3.0;")
if not hasattr(_qiskit, "transpile"):
    _qiskit.transpile = MagicMock(return_value=MagicMock())
if not hasattr(_qiskit, "QuantumCircuit"):
    _qiskit.QuantumCircuit = MagicMock(return_value=MagicMock())

_aer = sys.modules["qiskit_aer"]
if not hasattr(_aer, "AerSimulator"):
    _mock_backend = MagicMock()
    _mock_job = MagicMock()
    _mock_result = MagicMock()
    _mock_result.get_counts = MagicMock(return_value={"0000": 1})
    _mock_job.result = MagicMock(return_value=_mock_result)
    _mock_backend.run = MagicMock(return_value=_mock_job)
    _aer.AerSimulator = MagicMock(return_value=_mock_backend)


# ---------------------------------------------------------------------------
# transformers stub
# ---------------------------------------------------------------------------
_transformers = types.ModuleType("transformers")
_transformers.AutoTokenizer = MagicMock()
_transformers.AutoModelForCausalLM = MagicMock()
_transformers.BitsAndBytesConfig = MagicMock(return_value=MagicMock())
sys.modules.setdefault("transformers", _transformers)


# ---------------------------------------------------------------------------
# bitsandbytes / accelerate stubs
# ---------------------------------------------------------------------------
for _mod in ("bitsandbytes", "accelerate"):
    sys.modules.setdefault(_mod, types.ModuleType(_mod))


# ---------------------------------------------------------------------------
# pyquil stub
# ---------------------------------------------------------------------------
_pyquil = types.ModuleType("pyquil")
_pyquil.Program = MagicMock()
_pyquil.get_qc = MagicMock(return_value=MagicMock())
sys.modules.setdefault("pyquil", _pyquil)


# ---------------------------------------------------------------------------
# cirq stub
# ---------------------------------------------------------------------------
_cirq = types.ModuleType("cirq")
_cirq.LineQubit = MagicMock()
_cirq.Circuit = MagicMock(return_value=MagicMock())
_cirq.H = MagicMock()
_cirq.CNOT = MagicMock()
_cirq.measure = MagicMock()
_cirq.Simulator = MagicMock(return_value=MagicMock())
sys.modules.setdefault("cirq", _cirq)


# ---------------------------------------------------------------------------
# braket stub
# ---------------------------------------------------------------------------
for _mod in ("braket", "braket.aws", "braket.devices", "braket.ir", "braket.ir.openqasm"):
    sys.modules.setdefault(_mod, types.ModuleType(_mod))
_braket_oq = sys.modules["braket.ir.openqasm"]
if not hasattr(_braket_oq, "Program"):
    _braket_oq.Program = MagicMock()
_braket_devices = sys.modules["braket.devices"]
if not hasattr(_braket_devices, "LocalSimulator"):
    _mock_local_sim = MagicMock()
    _mock_local_sim.run = MagicMock(return_value=MagicMock())
    _braket_devices.LocalSimulator = MagicMock(return_value=_mock_local_sim)

