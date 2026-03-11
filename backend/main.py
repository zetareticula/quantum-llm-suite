
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from .gpu_dispatcher import gpu_inference
from .quantum_dispatcher import quantum_inference, warmup_quantum_backend
from .route_planner import RouteOption, TaskRequest, explain_decision, get_route_planner
from .scheduler import start_scheduler
from .utils import HF_MODELS, QUANTUM_BACKENDS, compute_benchmark, validate_api_key, warmup_model


load_dotenv()
logger = logging.getLogger(__name__)


app = FastAPI(
    title="Quantum LLMaaS API",
    description="OpenAPI-compliant API for quantum-accelerated LLM serving with exposed quantum quantization aspects.",
    version="1.4.0",
    openapi_url="/openapi.json",
)


class InferenceRequest(BaseModel):
    prompt: str
    quantum_provider: Optional[str] = "IBM"
    cloud_offering: Optional[str] = "AWS"
    model: Optional[str] = "Qwen/Qwen2.5-72B-Instruct"
    quantization_level: Optional[str] = "fp16"
    quantization_mode: Optional[str] = "classical"
    use_quantum: bool = True
    use_real_hardware: bool = False


class InferenceResponse(BaseModel):
    classical_output: str
    quantum_output: str
    benchmark: Dict[str, Any]


def get_api_key(api_key: Optional[str] = Header(default=None, alias="api-key")) -> str:
    if not api_key or not validate_api_key(api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key


@app.on_event("startup")
async def _startup() -> None:
    start_scheduler()

    warmup = os.getenv("WARMUP_ON_START", "0")
    if warmup not in {"1", "true", "True"}:
        return

    model_name = os.getenv("WARMUP_MODEL", "Qwen/Qwen2.5-72B-Instruct")
    quant_level = os.getenv("WARMUP_QUANT", "fp16")
    provider = os.getenv("WARMUP_PROVIDER", "Local")
    use_real = os.getenv("WARMUP_REAL", "0") in {"1", "true", "True"}

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, warmup_model, model_name, quant_level)
    await loop.run_in_executor(None, warmup_quantum_backend, provider, use_real)


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"ok": True, "models": HF_MODELS, "quantum_backends": QUANTUM_BACKENDS}


@app.post("/inference", response_model=InferenceResponse)
async def run_inference(request: InferenceRequest, api_key: str = Depends(get_api_key)) -> InferenceResponse:
    loop = asyncio.get_running_loop()

    class_task = loop.run_in_executor(
        None,
        gpu_inference,
        request.prompt,
        request.cloud_offering or "AWS",
        request.model or HF_MODELS[0],
        request.quantization_level or "fp16",
        request.quantization_mode or "classical",
    )

    if request.use_quantum:
        quant_task = loop.run_in_executor(
            None,
            quantum_inference,
            request.prompt,
            request.quantum_provider or "Local",
            bool(request.use_real_hardware),
            request.model or HF_MODELS[0],
            request.quantization_level or "fp16",
            request.quantization_mode or "classical",
        )
    else:
        async def _noop() -> Any:
            return ("Quantum disabled", 0.0)

        quant_task = _noop()

    (class_out, class_time), (quant_out, quant_time) = await asyncio.gather(class_task, quant_task)
    bench = compute_benchmark(class_out, quant_out, class_time, quant_time)

    return InferenceResponse(
        classical_output=class_out,
        quantum_output=quant_out,
        benchmark={
            "classical_seconds": bench.classical_seconds,
            "quantum_seconds": bench.quantum_seconds,
            "speedup": bench.speedup,
            "divergence": bench.divergence,
        },
    )


@app.get("/route-planner")
async def route_planner(
    api_key: str = Depends(get_api_key),
    planner: str = "greedy",
    priority: int = 0,
    max_latency_ms: Optional[int] = None,
    budget: Optional[float] = None,
) -> Dict[str, Any]:
    options = [
        RouteOption(name="classical_gpu", cost_per_1k_tokens=0.2, est_latency_ms=800, quality=0.8, capabilities={"classical"}),
        RouteOption(name="quantum_seeded", cost_per_1k_tokens=0.5, est_latency_ms=1800, quality=0.9, capabilities={"quantum"}),
    ]
    req = TaskRequest(
        prompt="route-planner",
        required_capabilities={"quantum"},
        budget=budget,
        max_latency_ms=max_latency_ms,
        priority=priority,
    )
    rp = get_route_planner(planner)
    decision = rp.choose_route(req, options)
    return explain_decision(req, decision)