"""
Stage 4: Parse and Validate API Responses

Reads each raw response, extracts the JSON payload (direct parse, code fence,
or balanced-brace extraction), validates against the POSAS observer schema,
runs plausibility checks (e.g. all-identical scores), and writes the outcome
to outputs/validated/{model}/{image_id}.json.

Schema: config/posas_schema.json

Expected v4 response shape:
    {"observer_scores": {
        "vascularity": <1-10>, "pigmentation": <1-10>, "thickness": <1-10>,
        "relief": <1-10>, "surface_area": <1-10>
    }}
"""

import csv
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import jsonschema

logger = logging.getLogger("posas")

OBSERVER_ITEMS = [
    "vascularity", "pigmentation", "thickness", "relief", "surface_area",
]

SCHEMA_PATH = "config/posas_schema.json"


def _load_schema() -> dict:
    return json.loads(Path(SCHEMA_PATH).read_text(encoding="utf-8"))


def _parse_json(raw_text: str) -> tuple:
    """
    Extract valid JSON from a model response.
    Returns (parsed_dict, method_name) or (None, None) on failure.
    """
    if raw_text is None:
        return None, None

    text = raw_text.strip()

    try:
        return json.loads(text), "direct_json"
    except json.JSONDecodeError:
        pass

    pattern = r"```(?:json)?\s*([\s\S]*?)```"
    match = re.search(pattern, text)
    if match:
        try:
            return json.loads(match.group(1).strip()), "code_fence"
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    if start != -1:
        depth = 0
        in_str = False
        esc = False
        for i, ch in enumerate(text[start:], start):
            if esc:
                esc = False
                continue
            if ch == "\\" and in_str:
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1]), "brace_extraction"
                    except json.JSONDecodeError:
                        break

    return None, None


def _validate_schema(parsed: dict, schema: dict) -> tuple:
    try:
        jsonschema.validate(instance=parsed, schema=schema)
        return True, None
    except jsonschema.ValidationError as e:
        return False, e.message
    except jsonschema.SchemaError as e:
        return False, f"Schema error: {e.message}"


def _plausibility_checks(parsed: dict) -> list:
    """Flag potentially unreliable scoring patterns."""
    flags = []
    scores_obj = parsed.get("observer_scores", {})
    scores = [
        scores_obj[item]
        for item in OBSERVER_ITEMS
        if item in scores_obj and isinstance(scores_obj[item], int)
    ]

    if not scores:
        return flags

    if len(set(scores)) == 1:
        flags.append(f"all_scores_identical: all={scores[0]}")
    if all(s == 1 for s in scores):
        flags.append("extreme_uniformity: all scores = 1")
    if all(s == 10 for s in scores):
        flags.append("extreme_uniformity: all scores = 10")

    return flags


def _validated_path(validated_dir: str, model_name: str, image_id: str) -> Path:
    return Path(validated_dir) / model_name / f"{image_id}.json"


def _save_validated(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _update_run_log_validation(run_log_path: str, validated_dir: str) -> None:
    path = Path(run_log_path)
    if not path.exists():
        return

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return

    validated_lookup: dict = {}
    for model_dir in sorted(Path(validated_dir).iterdir()):
        if not model_dir.is_dir():
            continue
        for vf in model_dir.glob("*.json"):
            data = json.loads(vf.read_text(encoding="utf-8"))
            key = (data["image_id"], data["model_name"])
            validated_lookup[key] = data["validation_passed"]

    seen: dict = {}
    for i, row in enumerate(rows):
        key = (row["image_id"], row["model_name"])
        if key not in seen:
            seen[key] = i
        else:
            prev_ts = rows[seen[key]].get("timestamp", "")
            curr_ts = row.get("timestamp", "")
            if curr_ts > prev_ts:
                seen[key] = i

    kept_indices = set(seen.values())
    deduped = [row for i, row in enumerate(rows) if i in kept_indices]
    deduped.sort(key=lambda r: (r["image_id"], r["model_name"]))

    removed = len(rows) - len(deduped)

    for row in deduped:
        key = (row["image_id"], row["model_name"])
        if key in validated_lookup:
            row["validation_passed"] = str(validated_lookup[key]).upper()

    fieldnames = rows[0].keys()
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(deduped)

    if removed:
        logger.info(f"  run_log: removed {removed} duplicate row(s)")


def run_stage4(settings: dict) -> None:
    raw_responses_dir = Path(settings["paths"]["raw_responses"])
    validated_dir     = settings["paths"]["validated"]
    run_log_path      = settings["paths"]["run_log"]

    logger.info("=" * 60)
    logger.info("STAGE 4: Validation")
    logger.info("=" * 60)

    if not raw_responses_dir.exists():
        logger.warning("No raw_responses folder found. Run Stage 3 first.")
        return

    schema = _load_schema()

    total        = 0
    passed       = 0
    failed_parse = 0
    failed_schema = 0
    flagged      = 0

    for model_dir in sorted(raw_responses_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        for raw_file in sorted(model_dir.glob("*.json")):
            image_id = raw_file.stem
            out_path = _validated_path(validated_dir, model_name, image_id)

            total += 1

            raw_data = json.loads(raw_file.read_text(encoding="utf-8"))

            # API error with no response text (e.g. content-moderation refusal)
            if raw_data.get("error") and not raw_data.get("raw_response_text"):
                result = {
                    "image_id": image_id,
                    "model_name": model_name,
                    "validation_passed": False,
                    "parse_method": None,
                    "schema_valid": False,
                    "plausibility_flags": [],
                    "parsed_scores": None,
                    "validation_timestamp": datetime.now(timezone.utc).isoformat(),
                    "error": raw_data.get("error"),
                }
                _save_validated(out_path, result)
                failed_parse += 1
                logger.warning(f"  {model_name}/{image_id}: API error (no response)")
                continue

            parsed, parse_method = _parse_json(raw_data.get("raw_response_text"))

            if parsed is None:
                result = {
                    "image_id": image_id,
                    "model_name": model_name,
                    "validation_passed": False,
                    "parse_method": None,
                    "schema_valid": False,
                    "plausibility_flags": [],
                    "parsed_scores": None,
                    "validation_timestamp": datetime.now(timezone.utc).isoformat(),
                    "error": "JSON parse failed",
                }
                _save_validated(out_path, result)
                failed_parse += 1
                logger.warning(f"  {model_name}/{image_id}: JSON parse failed")
                continue

            schema_valid, schema_error = _validate_schema(parsed, schema)
            if not schema_valid:
                result = {
                    "image_id": image_id,
                    "model_name": model_name,
                    "validation_passed": False,
                    "parse_method": parse_method,
                    "schema_valid": False,
                    "plausibility_flags": [],
                    "parsed_scores": parsed,
                    "validation_timestamp": datetime.now(timezone.utc).isoformat(),
                    "error": schema_error,
                }
                _save_validated(out_path, result)
                failed_schema += 1
                logger.warning(f"  {model_name}/{image_id}: schema invalid — {schema_error}")
                continue

            flags = _plausibility_checks(parsed)
            if flags:
                flagged += 1
                logger.warning(f"  {model_name}/{image_id}: plausibility flags: {flags}")

            result = {
                "image_id": image_id,
                "model_name": model_name,
                "validation_passed": True,
                "parse_method": parse_method,
                "schema_valid": True,
                "plausibility_flags": flags,
                "parsed_scores": parsed["observer_scores"],
                "validation_timestamp": datetime.now(timezone.utc).isoformat(),
                "error": None,
            }
            _save_validated(out_path, result)
            passed += 1
            logger.info(f"  {model_name}/{image_id}: OK  (parse: {parse_method})")

    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 4 COMPLETE")
    logger.info(f"Total responses:    {total}")
    logger.info(f"Passed validation:  {passed}")
    logger.info(f"Parse failures:     {failed_parse}")
    logger.info(f"Schema failures:    {failed_schema}")
    logger.info(f"Plausibility flags: {flagged}")
    logger.info("")
    logger.info("Updating run_log.csv with validation results...")
    _update_run_log_validation(run_log_path, validated_dir)
    logger.info("  run_log.csv updated.")
    logger.info("=" * 60)
