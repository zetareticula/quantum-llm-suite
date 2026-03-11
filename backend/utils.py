
from __future__ import annotations

import importlib
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


API_KEYS = {
    # In production, set `API_KEYS` or `API_KEY` via env.
    # This fallback keeps local dev from being blocked.
    "dev-key",
}


HF_MODELS = [
    "Qwen/Qwen2.5-72B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "mistralai/Mistral-Large-Instruct-2407",
    "THUDM/glm-4-9b-chat",
    "DeepSeek-AI/DeepSeek-V2-Chat",
]


QUANTUM_BACKENDS = ["IBM", "Rigetti", "Google Quantum AI", "IonQ", "Local", "AWS Braket"]


def validate_api_key(api_key: str) -> bool:
    if not api_key:
        return False
    env_keys = os.getenv("API_KEYS")
    if env_keys:
        allowed = {k.strip() for k in env_keys.split(",") if k.strip()}
        return api_key in allowed
    env_key = os.getenv("API_KEY")
    if env_key:
        return api_key == env_key
    return api_key in API_KEYS


@dataclass(frozen=True)
class Benchmark:
    classical_seconds: float
    quantum_seconds: float
    speedup: float
    divergence: float


def compute_benchmark(classical_output: str, quantum_output: str, classical_seconds: float, quantum_seconds: float) -> Benchmark:
    classical_seconds = float(max(1e-9, classical_seconds))
    quantum_seconds = float(max(1e-9, quantum_seconds))
    speedup = classical_seconds / quantum_seconds
    # A tiny, explainable divergence metric (0..1-ish): normalized length difference.
    clen = max(1, len(classical_output or ""))
    qlen = max(1, len(quantum_output or ""))
    divergence = abs(clen - qlen) / float(max(clen, qlen))
    return Benchmark(
        classical_seconds=classical_seconds,
        quantum_seconds=quantum_seconds,
        speedup=speedup,
        divergence=divergence,
    )


def _lazy_import(module_name: str) -> Any:
    return importlib.import_module(module_name)


_MODEL_CACHE: Dict[Tuple[str, str], Any] = {}
_TOKENIZER_CACHE: Dict[str, Any] = {}
_CACHE_LOCK = threading.Lock()


def get_quantization_config(quant_level: str) -> Optional[Any]:
    """Returns a BitsAndBytesConfig for int8/int4, otherwise None.

    Lazy-imports transformers so importing the backend doesn't immediately allocate HF machinery.
    """
    quant_level = (quant_level or "fp16").lower()
    if quant_level not in {"fp16", "int8", "int4"}:
        quant_level = "fp16"
    if quant_level == "fp16":
        return None
    transformers = _lazy_import("transformers")
    bnb = getattr(transformers, "BitsAndBytesConfig")
    if quant_level == "int8":
        return bnb(load_in_8bit=True)
    return bnb(load_in_4bit=True)


def get_tokenizer(model_name: str) -> Any:
    model_name = str(model_name)
    with _CACHE_LOCK:
        tok = _TOKENIZER_CACHE.get(model_name)
    if tok is not None:
        return tok

    transformers = _lazy_import("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    with _CACHE_LOCK:
        _TOKENIZER_CACHE[model_name] = tokenizer
    return tokenizer


def get_model(model_name: str, quant_level: str) -> Any:
    """Loads and caches HF models by (model_name, quant_level).

    Notes on memory:
    - We avoid importing transformers/torch until the first inference call.
    - We cache a single instance per key to avoid repeated huge allocations.
    """
    model_name = str(model_name)
    quant_level = (quant_level or "fp16").lower()
    key = (model_name, quant_level)

    with _CACHE_LOCK:
        model = _MODEL_CACHE.get(key)
    if model is not None:
        return model

    transformers = _lazy_import("transformers")
    quant_cfg = get_quantization_config(quant_level)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_cfg,
        device_map="auto",
    )
    model.eval()
    with _CACHE_LOCK:
        _MODEL_CACHE[key] = model
    return model


def warmup_model(model_name: str, quant_level: str = "fp16") -> Dict[str, Any]:
    """Preloads tokenizer + model and returns timing metrics."""
    t0 = time.time()
    tok = get_tokenizer(model_name)
    t1 = time.time()
    model = get_model(model_name, quant_level)
    t2 = time.time()
    return {
        "tokenizer_seconds": t1 - t0,
        "model_seconds": t2 - t1,
        "total_seconds": t2 - t0,
        "model": getattr(model, "__class__", type(model)).__name__,
        "tokenizer": getattr(tok, "__class__", type(tok)).__name__,
    }