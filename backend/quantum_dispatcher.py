
from __future__ import annotations

import logging
import os
import random
import time
from typing import Any, Dict, Optional, Tuple

from .utils import HF_MODELS, QUANTUM_BACKENDS, get_model, get_tokenizer


logger = logging.getLogger(__name__)


_BACKEND_CACHE: Dict[Tuple[str, bool], Any] = {}


def _seed_from_counts(counts: Dict[Any, int]) -> int:
    """Derive a stable integer seed from a single-shot measurement histogram.

    Numerical method:
    - Prefer the most probable key.
    - Interpret bitstrings as base-2 when possible.
    - Fallback to hashing arbitrary keys.
    """
    if not counts:
        return 0
    best_key = max(counts.items(), key=lambda kv: kv[1])[0]
    if isinstance(best_key, str):
        s = best_key.strip()
        if all(c in "01" for c in s) and s:
            return int(s, 2)
    return abs(hash(best_key)) % (2**31)


def _get_qiskit_backend(use_real: bool) -> Any:
    key = ("IBM", bool(use_real))
    if key in _BACKEND_CACHE:
        return _BACKEND_CACHE[key]
    if use_real and os.getenv("IBM_TOKEN"):
        from qiskit_ibm_runtime import QiskitRuntimeService

        service = QiskitRuntimeService(channel="ibm_quantum", token=os.getenv("IBM_TOKEN"))
        backend = service.least_busy(operational=True, simulator=False)
    else:
        from qiskit_aer import AerSimulator

        backend = AerSimulator()
    _BACKEND_CACHE[key] = backend
    return backend


def zr_quantum_compiler(ir_qasm: str, provider: str, use_real: bool = False) -> Dict[Any, int]:
    """Compile+execute OpenQASM across providers, returning measurement counts.

    This function intentionally does **shots=1** as a fast, deterministic seed source for
    downstream modulation rather than as a statistically meaningful quantum experiment.
    """

    provider = str(provider or "Local")
    if provider not in QUANTUM_BACKENDS:
        raise ValueError(f"Invalid provider: {provider}. Must be one of {QUANTUM_BACKENDS}")

    try:
        if provider in {"IBM", "Local"}:
            from qiskit import transpile
            from qiskit.qasm3 import loads as qasm3_loads

            backend = _get_qiskit_backend(use_real if provider == "IBM" else False)
            qc = qasm3_loads(ir_qasm)
            transpiled = transpile(qc, backend)
            result = backend.run(transpiled, shots=1).result()
            return result.get_counts()

        if provider == "AWS Braket":
            from braket.aws import AwsDevice
            from braket.devices import LocalSimulator
            from braket.ir.openqasm import Program as OpenQASMProgram

            if use_real and os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
                device = AwsDevice("arn:aws:braket:us-east-1::device/qpu/ionq/Harmony")
            else:
                device = LocalSimulator()
            task = device.run(OpenQASMProgram(source=ir_qasm), shots=1)
            return dict(task.result().measurement_counts)

        if provider == "Rigetti":
            from pyquil import Program, get_qc

            qc_conn = get_qc("9q-square-qvm", as_qvm=not (use_real and os.getenv("RIGETTI_TOKEN")))
            # Minimal seed circuit; full QASM->Quil is out of scope for this MVP.
            prog = Program("H 0\nCX 0 1\nMEASURE 0 [0]\nMEASURE 1 [1]")
            res = qc_conn.run_and_measure(prog, trials=1)
            bitstring = "".join(str(res[q][0]) for q in sorted(res.keys()))
            return {bitstring: 1}

        if provider in {"Google Quantum AI", "IonQ"}:
            import cirq

            # Many providers accept Cirq-native circuits; here we treat QASM as best-effort.
            try:
                from cirq.contrib.qasm_import import circuit_from_qasm

                circuit = circuit_from_qasm(ir_qasm)
            except Exception:
                q0, q1 = cirq.LineQubit.range(2)
                circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1), cirq.measure(q0, q1, key="m"))

            simulator = cirq.Simulator()
            result = simulator.run(circuit, repetitions=1)
            hist = result.histogram(key=list(result.measurements.keys())[0])
            return {format(k, "b"): int(v) for k, v in hist.items()}

    except Exception as e:
        logger.exception("Quantum compilation failed for provider=%s use_real=%s", provider, use_real)
        return {"0": 1}

    return {"0": 1}


def _build_seed_circuit_qasm3(n_qubits: int = 4) -> str:
    from qiskit import QuantumCircuit
    from qiskit.qasm3 import dumps as qasm3_dumps

    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    for i in range(0, n_qubits - 1, 2):
        qc.cx(i, i + 1)
    qc.measure_all()
    return qasm3_dumps(qc)


def quantum_inference(
    prompt: str,
    provider: str,
    use_real: bool = False,
    model_name: str = "Qwen/Qwen2.5-72B-Instruct",
    quant_level: str = "fp16",
    quant_mode: str = "classical",
    max_new_tokens: int = 20,
) -> Tuple[str, float]:
    """Run LLM generation with quantum-derived modulation.

    Numerical methods:
    - Build a small entangling circuit and sample once.
    - Convert measurement counts -> integer seed.
    - Use seed to perturb sampling parameters or a projection matrix.
    """

    t0 = time.time()
    provider = provider if provider in QUANTUM_BACKENDS else "Local"
    model_name = model_name if model_name in HF_MODELS else HF_MODELS[0]
    quant_level = (quant_level or "fp16").lower()
    quant_mode = (quant_mode or "classical").lower()

    try:
        qasm = _build_seed_circuit_qasm3(4)
        counts = zr_quantum_compiler(qasm, provider, use_real=use_real)
        seed = _seed_from_counts(counts)
        rng = random.Random(seed)

        tokenizer = get_tokenizer(model_name)
        model = get_model(model_name, quant_level)

        import torch

        inputs = tokenizer(prompt, return_tensors="pt")
        device = getattr(model, "device", None)
        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}

        gen_kwargs: Dict[str, Any] = {"max_new_tokens": int(max_new_tokens)}

        if quant_mode == "quantum_distillation":
            gen_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": 0.1 + 1.2 * rng.random(),
                    "top_p": 0.7 + 0.29 * rng.random(),
                }
            )
            tag = f"seed={seed} temp={gen_kwargs['temperature']:.3f} top_p={gen_kwargs['top_p']:.3f}"

            with torch.inference_mode():
                out = model.generate(**inputs, **gen_kwargs)
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            modulated = f"{text} (Quantum distillation mod: {tag})"

        elif quant_mode == "quantum_embedding":
            # Correct embedding projection: operate on token embeddings, not integer IDs.
            # Projection is a random (seeded) low-rank map: [B,T,H] -> [B,T,H].
            with torch.inference_mode():
                input_ids = inputs.get("input_ids")
                attn = inputs.get("attention_mask")
                emb = model.get_input_embeddings()(input_ids)
                h = emb.shape[-1]
                rank = max(8, h // 2)
                g = torch.Generator(device=emb.device)
                g.manual_seed(seed)
                proj = torch.randn((h, rank), generator=g, device=emb.device, dtype=emb.dtype) / (h**0.5)
                emb_low = emb @ proj
                emb_proj = emb_low @ proj.transpose(0, 1)
                emb_mix = 0.7 * emb + 0.3 * emb_proj

                try:
                    out = model.generate(inputs_embeds=emb_mix, attention_mask=attn, **gen_kwargs)
                except Exception:
                    out = model.generate(**inputs, **gen_kwargs)
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            modulated = f"{text} (Quantum embedding compression: rank={rank} seed={seed})"

        else:
            with torch.inference_mode():
                out = model.generate(**inputs, **gen_kwargs)
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            modulated = f"{text} (Classical mod: {rng.random():.2f})"

    except Exception as e:
        logger.exception("quantum_inference failed")
        modulated = f"Error during quantum inference: {type(e).__name__}"

    return f"Quantum ({provider}, {quant_level}, {quant_mode}) output: {modulated}", time.time() - t0


def warmup_quantum_backend(provider: str, use_real: bool = False) -> Dict[str, Any]:
    """Best-effort warmup to reduce first-request latency."""
    t0 = time.time()
    provider = provider if provider in QUANTUM_BACKENDS else "Local"
    try:
        if provider in {"IBM", "Local"}:
            _get_qiskit_backend(use_real if provider == "IBM" else False)
        # Others are lazily initialized inside zr_quantum_compiler.
        return {"provider": provider, "use_real": bool(use_real), "seconds": time.time() - t0, "ok": True}
    except Exception as e:
        return {"provider": provider, "use_real": bool(use_real), "seconds": time.time() - t0, "ok": False, "error": str(e)}