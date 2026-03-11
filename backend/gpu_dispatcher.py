
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Tuple

from .utils import HF_MODELS, get_model, get_tokenizer


logger = logging.getLogger(__name__)


def gpu_inference(
    prompt: str,
    cloud: str,
    model_name: str = "Qwen/Qwen2.5-72B-Instruct",
    quant_level: str = "fp16",
    quant_mode: str = "classical",
    max_new_tokens: int = 20,
) -> Tuple[str, float]:
    t0 = time.time()

    model_name = model_name if model_name in HF_MODELS else HF_MODELS[0]
    quant_level = (quant_level or "fp16").lower()
    quant_mode = (quant_mode or "classical").lower()

    try:
        tokenizer = get_tokenizer(model_name)
        model = get_model(model_name, quant_level)

        import torch

        inputs = tokenizer(prompt, return_tensors="pt")
        device = getattr(model, "device", None)
        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}

        gen_kwargs: Dict[str, Any] = {"max_new_tokens": int(max_new_tokens)}

        if quant_mode == "quantum_distillation":
            # Classical approximation: fixed sampling params.
            gen_kwargs.update({"do_sample": True, "temperature": 0.7, "top_p": 0.9})

        elif quant_mode == "quantum_embedding":
            # Classical approximation: deterministic low-rank projection on embeddings.
            with torch.inference_mode():
                input_ids = inputs.get("input_ids")
                attn = inputs.get("attention_mask")
                emb = model.get_input_embeddings()(input_ids)
                h = emb.shape[-1]
                rank = max(8, h // 2)
                g = torch.Generator(device=emb.device)
                g.manual_seed(0)
                proj = torch.randn((h, rank), generator=g, device=emb.device, dtype=emb.dtype) / (h**0.5)
                emb_low = emb @ proj
                emb_proj = emb_low @ proj.transpose(0, 1)
                emb_mix = 0.7 * emb + 0.3 * emb_proj
                try:
                    out = model.generate(inputs_embeds=emb_mix, attention_mask=attn, **gen_kwargs)
                except Exception:
                    out = model.generate(**inputs, **gen_kwargs)
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            return (
                f"Classical ({cloud}, {quant_level}, {quant_mode}) output: {text} (Embedding sim: rank={rank})",
                time.time() - t0,
            )

        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        suffix = ""
        if quant_mode == "quantum_distillation":
            suffix = " (Classical distillation sim)"
        return f"Classical ({cloud}, {quant_level}, {quant_mode}) output: {text}{suffix}", time.time() - t0

    except Exception as e:
        logger.exception("gpu_inference failed")
        return f"Classical ({cloud}, {quant_level}, {quant_mode}) output: Error: {type(e).__name__}", time.time() - t0