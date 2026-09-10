"""
Stage 3: API Dispatch

Sends each preprocessed image to every configured model, saves the raw
response as JSON per (model, image), appends a row to run_log.csv, and
tracks token usage + estimated cost.

Runs each image's model calls in parallel via ThreadPoolExecutor.
Already-completed (raw + validated) calls are skipped so the stage is
resumable after a partial failure.
"""

import base64
import csv
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from src.adapters.openai_adapter import OpenAIAdapter
from src.adapters.anthropic_adapter import AnthropicAdapter
from src.adapters.gemini_adapter import GeminiAdapter
from src.adapters.openrouter_adapter import OpenRouterAdapter
from src.stage1_ingest import read_manifest
from src.utils.retry_handler import retry_api_call, APICallError

logger = logging.getLogger("posas")

_ADAPTER_MAP = {
    "openai_adapter": OpenAIAdapter,
    "anthropic_adapter": AnthropicAdapter,
    "gemini_adapter": GeminiAdapter,
    "openrouter_adapter": OpenRouterAdapter,
}

# Per-provider image-token estimates (approximate — used only in the dry-run
# cost preview; actual usage is captured after each call from the API response).
_IMAGE_TOKEN_ESTIMATES = {
    "openai_adapter": 765,
    "anthropic_adapter": 1590,
    "gemini_adapter": 258,
    "openrouter_adapter": 1200,
}

_log_lock = threading.Lock()

RUN_LOG_COLUMNS = [
    "image_id", "model_name", "model_string", "timestamp", "prompt_version",
    "response_time_seconds", "input_tokens", "output_tokens",
    "estimated_cost_usd", "http_status", "retry_count",
    "validation_passed", "error_message", "raw_response_file",
]


def _load_prompt(prompt_path: str) -> str:
    path = Path(prompt_path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return path.read_text(encoding="utf-8").strip()


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _raw_response_path(raw_responses_dir: str, model_name: str, image_id: str) -> Path:
    return Path(raw_responses_dir) / model_name / f"{image_id}.json"


def _save_raw_response(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _append_run_log(log_path: str, row: dict) -> None:
    path = Path(log_path)
    write_header = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RUN_LOG_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _get_active_models(settings: dict) -> list:
    active = []
    for model_name, cfg in settings["models"].items():
        key = os.environ.get(cfg["env_key"])
        if not key:
            logger.warning(
                f"Skipping {model_name}: {cfg['env_key']} not set in environment"
            )
            continue
        active.append((model_name, cfg))
    return active


def _dry_run_estimate(active_models: list, manifest: list, prompt: str,
                      prompt_version: str, auto_confirm: bool = False) -> None:
    import tiktoken
    try:
        enc = tiktoken.encoding_for_model("gpt-4o")
        prompt_tokens = len(enc.encode(prompt))
    except Exception:
        prompt_tokens = 60

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"DRY RUN — ESTIMATED COSTS  [prompt {prompt_version}]")
    logger.info("=" * 60)
    logger.info(f"Images to process:  {len(manifest)}")
    logger.info(f"Models active:      {len(active_models)}")
    logger.info(f"Prompt tokens:      ~{prompt_tokens}")
    logger.info("")
    logger.info(f"{'Model':<16} {'Est input tok':>14} {'Est cost/img':>13} {'Est total':>10}")
    logger.info("-" * 58)

    total_cost = 0.0
    for model_name, cfg in active_models:
        adapter_key = cfg["adapter"]
        img_tokens = _IMAGE_TOKEN_ESTIMATES.get(adapter_key, 500)
        input_tokens = prompt_tokens + img_tokens
        output_tokens = 30
        cost_per_call = (
            (input_tokens / 1_000_000) * cfg["pricing_input_per_1m"]
            + (output_tokens / 1_000_000) * cfg["pricing_output_per_1m"]
        )
        model_total = cost_per_call * len(manifest)
        total_cost += model_total
        logger.info(
            f"{model_name:<16} {input_tokens:>14,} {cost_per_call:>12.4f}$ {model_total:>9.4f}$"
        )

    logger.info("-" * 58)
    logger.info(f"{'TOTAL':<16} {'':>14} {'':>13} {total_cost:>9.4f}$")
    logger.info("=" * 60)

    extrapolated = total_cost * (100 / len(manifest)) if len(manifest) > 0 else 0
    logger.info(f"Extrapolated cost for 100 images: ~${extrapolated:.2f}")
    logger.info("")

    if auto_confirm:
        logger.info("Auto-confirmed (--yes flag). Proceeding.")
        return
    try:
        confirm = input("Proceed with API calls? [y/N]: ").strip().lower()
        if confirm != "y":
            logger.info("Aborted by user. No API calls made.")
            raise SystemExit(0)
    except EOFError:
        logger.info("Non-interactive mode — proceeding automatically.")


def _fmt_time(seconds: float) -> str:
    s = int(round(seconds))
    if s >= 3600:
        return f"{s // 3600}h {(s % 3600) // 60}m"
    if s >= 60:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s}s"


def _tok_str(tokens: dict) -> str:
    in_t  = tokens["input_tokens"]
    out_t = tokens["output_tokens"]
    think = tokens.get("reasoning_tokens")
    base  = f"{in_t:,} in -> {out_t:,} out"
    if think:
        return f"{base}  ({think:,} thinking)"
    return base


def _compute_cost(tokens: dict, cfg: dict) -> float:
    return (
        (tokens["input_tokens"] / 1_000_000) * cfg["pricing_input_per_1m"]
        + (tokens["output_tokens"] / 1_000_000) * cfg["pricing_output_per_1m"]
    )


def _dispatch_model(
    *,
    model_name: str,
    model_idx: int,
    total_models: int,
    cfg: dict,
    image_id: str,
    image_b64: str,
    prompt: str,
    prompt_version: str,
    raw_responses_dir: str,
    validated_dir: str,
    retry_cfg: dict,
    run_log_path: str,
) -> dict:
    try:
        out_path    = _raw_response_path(raw_responses_dir, model_name, image_id)
        is_thinking = cfg.get("is_thinking", False)
        think_hint  = "  [thinking model — may take 60-120s]" if is_thinking else ""
        model_label = f"[{model_idx}/{total_models}] {model_name:<14}"

        if out_path.exists():
            validated_path = Path(validated_dir) / model_name / f"{image_id}.json"
            if validated_path.exists():
                try:
                    v = json.loads(validated_path.read_text(encoding="utf-8"))
                except Exception:
                    v = {}
                if v.get("validation_passed"):
                    with _log_lock:
                        logger.info(f"  {model_label}  SKIP   already validated")
                    return {"model_name": model_name, "status": "skip"}
                reason = v.get("error") or "schema/parse error"
                with _log_lock:
                    logger.info(f"  {model_label}  RETRY  prev failed: {reason}")
            else:
                with _log_lock:
                    logger.info(f"  {model_label}  SKIP   raw exists, not yet validated")
                return {"model_name": model_name, "status": "skip"}

        adapter_class = _ADAPTER_MAP[cfg["adapter"]]
        adapter       = adapter_class(cfg)
        try:
            adapter.authenticate()
        except ValueError as e:
            with _log_lock:
                logger.error(f"  {model_label}  AUTH FAILED — {e}")
            return {"model_name": model_name, "status": "fail", "cost": 0.0, "elapsed": 0.0}

        payload   = adapter.prepare_payload(image_b64, prompt)
        timestamp = datetime.now(timezone.utc).isoformat()

        with _log_lock:
            logger.info(f"  {model_label}  --> {cfg['model_string']}{think_hint}")

        log_row = {
            "image_id": image_id, "model_name": model_name,
            "model_string": cfg["model_string"],
            "timestamp": timestamp,
            "prompt_version": prompt_version,
            "response_time_seconds": "", "input_tokens": "", "output_tokens": "",
            "estimated_cost_usd": "", "http_status": "", "retry_count": 0,
            "validation_passed": "", "error_message": "",
            "raw_response_file": str(out_path),
        }

        t_start = time.time()
        try:
            raw = retry_api_call(
                call_fn=lambda p=payload, a=adapter: a.send_request(p),
                model_name=model_name,
                max_retries=retry_cfg["max_retries"],
                base_delay=retry_cfg["base_delay_seconds"],
                jitter_max=retry_cfg["jitter_max_seconds"],
                retry_on=retry_cfg["retry_on_status_codes"],
                do_not_retry_on=retry_cfg["do_not_retry_on"],
            )
            elapsed       = round(time.time() - t_start, 2)
            response_text = adapter.extract_response_text(raw)
            tokens        = adapter.get_token_usage(raw)
            call_cost     = _compute_cost(tokens, cfg)

            _save_raw_response(out_path, {
                "image_id": image_id, "model_name": model_name,
                "model_string": cfg["model_string"],
                "timestamp": timestamp,
                "prompt_version": prompt_version,
                "response_time_seconds": elapsed,
                "input_tokens": tokens["input_tokens"],
                "output_tokens": tokens["output_tokens"],
                "reasoning_tokens": tokens.get("reasoning_tokens"),
                "estimated_cost_usd": call_cost,
                "http_status": 200, "retry_count": 0,
                "raw_response_text": response_text, "error": None,
            })

            log_row.update({
                "response_time_seconds": elapsed,
                "input_tokens": tokens["input_tokens"],
                "output_tokens": tokens["output_tokens"],
                "estimated_cost_usd": call_cost,
                "http_status": 200,
            })
            with _log_lock:
                logger.info(
                    f"  {model_label}  OK    "
                    f"{_fmt_time(elapsed):>7}  |  {_tok_str(tokens)}  |  "
                    f"${call_cost:.4f}"
                )
                _append_run_log(run_log_path, log_row)

            return {"model_name": model_name, "status": "ok",
                    "cost": call_cost, "elapsed": elapsed, "tokens": tokens}

        except APICallError as e:
            elapsed = round(time.time() - t_start, 2)
            _save_raw_response(out_path, {
                "image_id": image_id, "model_name": model_name,
                "model_string": cfg["model_string"], "timestamp": timestamp,
                "prompt_version": prompt_version,
                "response_time_seconds": elapsed,
                "input_tokens": 0, "output_tokens": 0, "estimated_cost_usd": 0,
                "http_status": e.status_code, "retry_count": e.retries,
                "raw_response_text": None, "error": str(e),
            })
            log_row.update({
                "response_time_seconds": elapsed,
                "http_status": e.status_code,
                "retry_count": e.retries,
                "error_message": str(e),
            })
            with _log_lock:
                logger.error(
                    f"  {model_label}  FAIL  "
                    f"{_fmt_time(elapsed):>7}  |  HTTP {e.status_code}  |  {e}"
                )
                _append_run_log(run_log_path, log_row)
            return {"model_name": model_name, "status": "fail",
                    "cost": 0.0, "elapsed": elapsed}

        except Exception as e:
            elapsed = round(time.time() - t_start, 2)
            log_row.update({
                "response_time_seconds": elapsed,
                "http_status": "unknown", "error_message": str(e),
            })
            with _log_lock:
                logger.error(
                    f"  {model_label}  FAIL  "
                    f"{_fmt_time(elapsed):>7}  |  {type(e).__name__}: {e}"
                )
                _append_run_log(run_log_path, log_row)
            return {"model_name": model_name, "status": "fail",
                    "cost": 0.0, "elapsed": elapsed}

    except Exception as e:
        with _log_lock:
            logger.error(f"  [{model_name}] WORKER EXCEPTION: {type(e).__name__}: {e}")
        return {"model_name": model_name, "status": "fail", "cost": 0.0, "elapsed": 0.0}


def run_stage3(settings: dict) -> None:
    # Load API keys from .env in project root, then fall back to shell environment.
    load_dotenv()

    manifest_path     = settings["paths"]["manifest"]
    raw_responses_dir = settings["paths"]["raw_responses"]
    run_log_path      = settings["paths"]["run_log"]
    prompt_path       = settings["paths"]["prompt_file"]
    prompt_version    = settings.get("prompt_version", "v4")
    pilot             = settings["pilot"]["enabled"]
    pilot_max         = settings["pilot"]["max_images"]
    retry_cfg         = settings["retry"]

    full_manifest = read_manifest(manifest_path)
    manifest = [r for r in full_manifest if r.get("processed_path")]
    if pilot:
        manifest = manifest[:pilot_max]

    image_filter = settings.get("image_filter")
    if image_filter:
        manifest = [r for r in manifest if r["original_filename"] in image_filter]
        if not manifest:
            logger.warning("No images matched --images filter. Check filenames.")
            return

    if not manifest:
        logger.warning("No preprocessed images found. Run Stage 2 first.")
        return

    prompt        = _load_prompt(prompt_path)
    active_models = _get_active_models(settings)
    total_images  = len(manifest)
    total_models  = len(active_models)
    total_calls   = total_images * total_models

    if not active_models:
        logger.error("No models available — check API keys in config/.env")
        return

    logger.info("=" * 64)
    logger.info("STAGE 3: API Dispatch")
    logger.info("=" * 64)
    logger.info(f"  Images  : {total_images}  ({manifest[0]['image_id']} — {manifest[-1]['image_id']})")
    logger.info(f"  Models  : {total_models}  [{' | '.join(m for m, _ in active_models)}]")
    logger.info(f"  Calls   : {total_calls}  total  ({total_models} models x {total_images} images)")
    logger.info(f"  Prompt  : {Path(prompt_path).name}")
    logger.info(f"  Outputs : {raw_responses_dir}")
    logger.info(f"  Mode    : parallel ({total_models} models/image via ThreadPoolExecutor)")
    logger.info("=" * 64)

    _dry_run_estimate(active_models, manifest, prompt, prompt_version,
                      auto_confirm=settings.get("auto_confirm", False))

    ok_calls        = 0
    failed_calls    = 0
    total_spent     = 0.0
    pipeline_start  = time.time()
    per_model_stats = {
        m: {"ok": 0, "fail": 0, "times": [], "costs": []}
        for m, _ in active_models
    }

    for img_idx, row in enumerate(manifest, start=1):
        image_id  = row["image_id"]
        orig_name = row["original_filename"]

        logger.info("")
        logger.info("-" * 64)
        logger.info(f"  IMAGE {img_idx}/{total_images}  |  {image_id}  |  {orig_name}")
        logger.info("-" * 64)

        try:
            image_b64 = _encode_image(row["processed_path"])
        except Exception as e:
            logger.error(f"  Cannot read image {row['processed_path']}: {e}")
            continue

        image_start = time.time()
        image_ok    = 0
        image_fail  = 0
        image_cost  = 0.0

        with ThreadPoolExecutor(max_workers=total_models) as executor:
            futures = {
                executor.submit(
                    _dispatch_model,
                    model_name=model_name,
                    model_idx=mi,
                    total_models=total_models,
                    cfg=cfg,
                    image_id=image_id,
                    image_b64=image_b64,
                    prompt=prompt,
                    prompt_version=prompt_version,
                    raw_responses_dir=raw_responses_dir,
                    validated_dir=settings["paths"]["validated"],
                    retry_cfg=retry_cfg,
                    run_log_path=run_log_path,
                ): model_name
                for mi, (model_name, cfg) in enumerate(active_models, start=1)
            }

            for future in as_completed(futures):
                result = future.result()
                mn     = result["model_name"]
                status = result["status"]

                if status == "ok":
                    cost    = result["cost"]
                    elapsed = result["elapsed"]
                    image_cost  += cost
                    image_ok    += 1
                    ok_calls    += 1
                    total_spent += cost
                    per_model_stats[mn]["ok"] += 1
                    per_model_stats[mn]["times"].append(elapsed)
                    per_model_stats[mn]["costs"].append(cost)
                elif status == "fail":
                    image_fail   += 1
                    failed_calls += 1
                    per_model_stats[mn]["fail"] += 1

        image_elapsed    = time.time() - image_start
        pipeline_elapsed = time.time() - pipeline_start
        avg_per_image    = pipeline_elapsed / img_idx
        remaining        = avg_per_image * (total_images - img_idx)
        eta_str          = f"  |  ETA: ~{_fmt_time(remaining)}" if img_idx < total_images else ""

        logger.info("")
        logger.info(
            f"  IMAGE {img_idx}/{total_images} DONE  |  "
            f"{image_ok}/{total_models} OK  "
            f"{'(' + str(image_fail) + ' failed)  ' if image_fail else ''}"
            f"|  image cost: ${image_cost:.4f}  |  image time: {_fmt_time(image_elapsed)}"
            f"{eta_str}"
        )
        logger.info(
            f"  CUMULATIVE: {ok_calls} calls OK  |  "
            f"total spent: ${total_spent:.4f}  |  "
            f"wall clock: {_fmt_time(pipeline_elapsed)}"
        )

    total_elapsed = time.time() - pipeline_start
    logger.info("")
    logger.info("=" * 64)
    logger.info("  STAGE 3 COMPLETE — Per-model breakdown")
    logger.info("=" * 64)
    logger.info(
        f"  {'Model':<14}  {'OK':>4}  {'Fail':>4}  "
        f"{'Avg time':>8}  {'$/call':>7}  {'Total $':>9}"
    )
    logger.info(f"  {'-'*14}  {'-'*4}  {'-'*4}  {'-'*8}  {'-'*7}  {'-'*9}")
    for model_name, cfg in active_models:
        st    = per_model_stats[model_name]
        avg_t = sum(st["times"]) / len(st["times"]) if st["times"] else 0
        avg_c = sum(st["costs"]) / len(st["costs"]) if st["costs"] else 0
        tot_c = sum(st["costs"])
        flag  = " [think]" if cfg.get("is_thinking", False) else ""
        logger.info(
            f"  {model_name:<14}  {st['ok']:>4}  {st['fail']:>4}  "
            f"{_fmt_time(avg_t):>8}  ${avg_c:>6.4f}  ${tot_c:>8.4f}{flag}"
        )
    logger.info(f"  {'-'*14}  {'-'*4}  {'-'*4}  {'-'*8}  {'-'*7}  {'-'*9}")
    logger.info(
        f"  {'TOTAL':<14}  {ok_calls:>4}  {failed_calls:>4}  "
        f"{'':>8}  {'':>7}  ${total_spent:>8.4f}"
    )
    logger.info("")
    logger.info(f"  [think] = reasoning model — output tokens include chain-of-thought")
    logger.info(f"  Wall time : {_fmt_time(total_elapsed)}")
    logger.info("=" * 64)
