from __future__ import annotations

import argparse
import inspect
import os
import re
from pathlib import Path
from typing import Any, Iterable, Tuple

import torch
from transformers import AutoModelForCausalLM as HFAutoModelForCausalLM
from transformers import AutoTokenizer as HFAutoTokenizer

try:
    from huggingface_hub import snapshot_download as hf_snapshot_download
except Exception:  # pragma: no cover - optional in some envs
    hf_snapshot_download = None

try:
    from modelscope import AutoModelForCausalLM as ModelScopeAutoModelForCausalLM
    from modelscope import AutoTokenizer as ModelScopeAutoTokenizer
    from modelscope import snapshot_download as ms_snapshot_download
except Exception:  # pragma: no cover - optional in some envs
    ModelScopeAutoModelForCausalLM = None
    ModelScopeAutoTokenizer = None
    ms_snapshot_download = None


MODEL_SOURCE_CHOICES = ("modelscope", "huggingface")
DEFAULT_MODEL_SOURCE = "modelscope"

_MODEL_ID_ALIASES = {
    "Qwen/Qwen3-8B-Instruct": "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B-Instruct": "Qwen/Qwen3-14B",
}

_MODELSCOPE_MODEL_ID_ALIASES = {
    "meta-llama/Llama-3.1-8B-Instruct": "LLM-Research/Meta-Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.1": "AI-ModelScope/Mistral-7B-Instruct-v0.1",
}


def add_model_source_arg(parser: argparse.ArgumentParser, default: str = DEFAULT_MODEL_SOURCE) -> None:
    parser.add_argument(
        "--model_source",
        choices=list(MODEL_SOURCE_CHOICES),
        default=default,
        help="Choose model loading backend.",
    )


def normalize_model_source(source: str) -> str:
    if source not in MODEL_SOURCE_CHOICES:
        raise ValueError(f"Unsupported model source {source!r}. Use one of: {MODEL_SOURCE_CHOICES}")
    return source


def resolve_model_id(model_name: str, source: str | None = None) -> str:
    resolved = _MODEL_ID_ALIASES.get(model_name, model_name)
    if source is None:
        return resolved

    normalized_source = normalize_model_source(source)
    if normalized_source == "modelscope":
        return _MODELSCOPE_MODEL_ID_ALIASES.get(resolved, resolved)
    return resolved


def resolve_transformer_lens_model_name(model_name: str, source: str | None = None) -> str:
    resolved = _MODEL_ID_ALIASES.get(model_name, model_name)
    if source is None:
        return resolved

    normalized_source = normalize_model_source(source)
    if normalized_source == "modelscope":
        for hf_name, ms_name in _MODELSCOPE_MODEL_ID_ALIASES.items():
            if resolved == ms_name:
                return hf_name
    return resolved


def resolve_model_artifact_tags(model_name: str) -> Tuple[str, ...]:
    resolved = str(resolve_model_id(model_name)).strip()
    base_name = Path(resolved.rstrip("/")).name or resolved
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", base_name).strip("-._")
    if not safe_name:
        return tuple()

    candidates = [safe_name]
    b_normalized = re.sub(r"(\d+)B\b", r"\1b", safe_name)
    if b_normalized != safe_name:
        candidates.append(b_normalized)
    lower_name = safe_name.lower()
    if lower_name not in candidates:
        candidates.append(lower_name)

    deduped = []
    seen = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            deduped.append(candidate)
    return tuple(deduped)


def _filter_supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs

    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _from_pretrained(loader_cls: Any, model_name_or_path: str, **kwargs: Any):
    filtered_kwargs = _filter_supported_kwargs(loader_cls.from_pretrained, kwargs)
    return loader_cls.from_pretrained(model_name_or_path, **filtered_kwargs)


def _snapshot_download_with_fallback(
    snapshot_fn: Any,
    model_name: str,
    *,
    cache_env_vars: Iterable[str],
    fallback_dir: str,
) -> str:
    cache_dir = next((os.environ.get(key) for key in cache_env_vars if os.environ.get(key)), None)
    try:
        if cache_dir:
            return snapshot_fn(model_name, cache_dir=cache_dir)
        return snapshot_fn(model_name)
    except PermissionError:
        fallback_cache = Path(fallback_dir)
        fallback_cache.mkdir(parents=True, exist_ok=True)
        return snapshot_fn(model_name, cache_dir=str(fallback_cache.resolve()))


def _find_existing_modelscope_cache_dir(model_name: str) -> str | None:
    resolved_model_id = resolve_model_id(model_name, source="modelscope")
    repo_path = Path(*resolved_model_id.split("/"))
    cache_roots = []
    if os.environ.get("MODELSCOPE_CACHE"):
        cache_roots.append(Path(os.environ["MODELSCOPE_CACHE"]).expanduser())
    cache_roots.append(Path.home() / ".cache" / "modelscope")
    cache_roots.append((Path.cwd() / ".cache" / "modelscope").resolve())

    seen: set[str] = set()
    for cache_root in cache_roots:
        cache_root_key = str(cache_root)
        if cache_root_key in seen:
            continue
        seen.add(cache_root_key)

        for prefix in (Path(), Path("models")):
            candidate = cache_root / prefix / repo_path
            if candidate.exists() and any(
                (candidate / filename).exists()
                for filename in ("config.json", "configuration.json")
            ):
                return str(candidate.expanduser().absolute())
    return None


def resolve_local_model_dir(model_name: str, source: str = DEFAULT_MODEL_SOURCE) -> str:
    source = normalize_model_source(source)
    aliased = resolve_model_id(model_name, source=source)

    path = Path(aliased)
    if path.exists():
        return str(path.expanduser().absolute())

    if source == "huggingface":
        if hf_snapshot_download is None:
            return aliased
        return _snapshot_download_with_fallback(
            hf_snapshot_download,
            aliased,
            cache_env_vars=("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"),
            fallback_dir=".cache/huggingface",
        )

    cached_model_dir = _find_existing_modelscope_cache_dir(model_name)
    if cached_model_dir is not None:
        return cached_model_dir

    if ms_snapshot_download is None:
        return aliased
    return _snapshot_download_with_fallback(
        ms_snapshot_download,
        aliased,
        cache_env_vars=("MODELSCOPE_CACHE",),
        fallback_dir=".cache/modelscope",
    )


def load_causal_lm(
    model_name: str,
    *,
    source: str = DEFAULT_MODEL_SOURCE,
    torch_dtype: Any = "auto",
    device_map: str = "auto",
):
    source = normalize_model_source(source)
    resolved_model_id = resolve_model_id(model_name, source=source)

    if source == "huggingface":
        tokenizer = _from_pretrained(HFAutoTokenizer, resolved_model_id, trust_remote_code=True)
        model = _from_pretrained(
            HFAutoModelForCausalLM,
            resolved_model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        return resolved_model_id, tokenizer, model

    if ModelScopeAutoTokenizer is None or ModelScopeAutoModelForCausalLM is None:
        raise ImportError(
            "modelscope is not installed, so --model_source modelscope is unavailable in this environment."
        )

    tokenizer = _from_pretrained(ModelScopeAutoTokenizer, resolved_model_id)
    model = _from_pretrained(
        ModelScopeAutoModelForCausalLM,
        resolved_model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    return resolved_model_id, tokenizer, model


def load_hooked_transformer(
    model_name: str,
    device: str = "cuda",
    dtype: str | None = None,
    *,
    source: str = DEFAULT_MODEL_SOURCE,
    error_context: str = "transformer_lens pipeline",
):
    try:
        from transformer_lens import HookedTransformer
        from transformer_lens import loading_from_pretrained as tl_loading
    except Exception as exc:
        raise ImportError(f"transformer_lens is required for {error_context}.") from exc

    resolved_local_dir = resolve_local_model_dir(model_name, source=source)
    if dtype is None:
        dtype = "float16" if str(device).startswith("cuda") else "float32"

    original_get_official_model_name = tl_loading.get_official_model_name

    def _patched_get_official_model_name(name: str) -> str:
        if Path(name).exists():
            return str(Path(name).expanduser().absolute())
        return original_get_official_model_name(name)

    tl_loading.get_official_model_name = _patched_get_official_model_name
    try:
        hf_tokenizer = _from_pretrained(
            HFAutoTokenizer,
            resolved_local_dir,
            trust_remote_code=True,
            local_files_only=True,
        )
        hf_model = _from_pretrained(
            HFAutoModelForCausalLM,
            resolved_local_dir,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=dtype,
            device_map=None,
        )
        model = HookedTransformer.from_pretrained(
            resolved_local_dir,
            hf_model=hf_model,
            tokenizer=hf_tokenizer,
            device=device,
            dtype=dtype,
            trust_remote_code=True,
        )
    except torch.OutOfMemoryError as exc:
        raise RuntimeError(
            f"Failed to load model {model_name!r} via {source} for {error_context} because device {device!r} ran out of memory. "
            "Try a less busy GPU, pass a different stage-specific device in the runner script, or free GPU memory before rerunning."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model {model_name!r} via {source} for {error_context}."
        ) from exc
    finally:
        tl_loading.get_official_model_name = original_get_official_model_name

    model.eval()
    return model


def resolve_true_false_token_ids(model) -> Tuple[int, int]:
    true_candidates = [" True", "True"]
    false_candidates = [" False", "False"]

    true_id = None
    false_id = None

    for item in true_candidates:
        ids = model.to_tokens(item, prepend_bos=False)[0].tolist()
        if ids:
            true_id = ids[-1]
            break

    for item in false_candidates:
        ids = model.to_tokens(item, prepend_bos=False)[0].tolist()
        if ids:
            false_id = ids[-1]
            break

    if true_id is None or false_id is None:
        raise RuntimeError("Unable to resolve True/False token IDs")

    return int(true_id), int(false_id)


def tokenizer_has_chat_template(tokenizer: Any) -> bool:
    return bool(
        getattr(tokenizer, "chat_template", None)
        or getattr(tokenizer, "default_chat_template", None)
    )


def render_generation_prompt(
    tokenizer: Any,
    prompt: str,
    *,
    enable_thinking: bool | None = False,
) -> str:
    if not hasattr(tokenizer, "apply_chat_template") or not tokenizer_has_chat_template(tokenizer):
        return prompt

    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking

    try:
        return str(tokenizer.apply_chat_template([{"role": "user", "content": prompt}], **kwargs))
    except TypeError:
        return str(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    except ValueError:
        return prompt


def resolve_model_prompt(
    model: Any,
    prompt: str,
    *,
    enable_thinking: bool | None = False,
) -> str:
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        return prompt
    return render_generation_prompt(tokenizer, prompt, enable_thinking=enable_thinking)


def to_tokens(model, prompt: str, prepend_bos: bool = True) -> torch.Tensor:
    toks = model.to_tokens(prompt, prepend_bos=prepend_bos)
    if toks.ndim != 2:
        raise ValueError(f"Expected [batch, pos] tokens, got shape {tuple(toks.shape)}")
    return toks
