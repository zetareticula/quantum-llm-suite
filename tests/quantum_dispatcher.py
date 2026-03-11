import pytest
from backend.quantum_dispatcher import zr_quantum_compiler, quantum_inference
from backend.utils import API_KEYS  # For mock

@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv('IBM_TOKEN', 'mock_token')  # For real fallback to sim

def test_zr_quantum_compiler_local():
    ir_qasm = """
    OPENQASM 3.0;
    include "stdgates.inc";
    qubit[2] q;
    h q[0];
    cx q[0], q[1];
    measure q;
    """
    counts = zr_quantum_compiler(ir_qasm, 'Local')
    assert isinstance(counts, dict)
    assert sum(counts.values()) == 1

def test_quantum_inference_classical(mock_env):
    out, time = quantum_inference("Test prompt", 'Local', False, 'THUDM/glm-4-9b-chat', 'fp16', 'classical')
    assert "output" in out
    assert time > 0

def test_quantum_inference_distillation(mock_env):
    out, time = quantum_inference("Test prompt", 'Local', False, 'THUDM/glm-4-9b-chat', 'int4', 'quantum_distillation')
    assert "distillation" in out
    assert time > 0

def test_quantum_inference_embedding(mock_env):
    out, time = quantum_inference("Test prompt", 'Local', False, 'THUDM/glm-4-9b-chat', 'int8', 'quantum_embedding')
    assert "embedding" in out
    assert time > 0

def test_invalid_provider():
    with pytest.raises(ValueError):
        zr_quantum_compiler("invalid", 'BadProvider')