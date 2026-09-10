# POSAS-MLLM Benchmark

Reproducible evaluation pipeline for multimodal large language models (MLLMs)
scoring clinical scar photographs against the **Patient and Observer Scar
Assessment Scale (POSAS) — Observer domain**.

This repository accompanies the study *"Multimodal Large Language Models for
Clinical Scar Assessment: Agreement, Bias, and Reproducibility"*. It contains
every stage of the evaluation pipeline from raw image ingestion to the
final `results_master.xlsx` used for statistical analysis.

The pipeline was originally used to evaluate five MLLMs on 107 blinded scar
photographs, but is not tied to that dataset — you can point it at any
folder of images and any subset of the four supported providers.

---

## What the pipeline does

Five stages, each runnable independently:

| Stage | Script                     | What happens                                                                                          |
|-------|----------------------------|-------------------------------------------------------------------------------------------------------|
| 1     | `stage1_ingest.py`         | Scans `data/raw_images/`, assigns sequential IDs (`IMG_001` …), computes SHA-256 hashes, writes manifest |
| 2     | `stage2_preprocess.py`     | Strips EXIF, auto-orients, resizes longest edge to 1024 px, pads to 1024×1024 JPEG-85                 |
| 3     | `stage3_dispatch.py`       | Sends each image to every configured model in parallel; retries on 429/5xx; saves raw responses       |
| 4     | `stage4_validate.py`       | Extracts JSON, validates against `config/posas_schema.json`, runs plausibility checks                 |
| 5     | `stage5_export.py`         | Writes `outputs/results_master.xlsx` (Main Results + Audit Log) and `results_master.csv`              |

The five MLLMs configured out of the box are:

| Key       | Model                 | Provider   | Adapter           |
|-----------|-----------------------|------------|-------------------|
| `chatgpt` | GPT-5.4               | OpenAI     | `openai_adapter`  |
| `claude`  | Claude Sonnet 5       | Anthropic  | `anthropic_adapter` |
| `gemini`  | Gemini 3.1 Pro        | Google     | `gemini_adapter`  |
| `qwen`    | Qwen3.7-Plus          | Alibaba    | `openrouter_adapter` |
| `glm`     | GLM-5V-Turbo          | Zhipu AI   | `openrouter_adapter` |

You can add, remove, or swap models by editing `config/settings.yaml`.

---

## Installation

Requires **Python 3.10 or newer**.

```bash
git clone https://github.com/<your-org>/posas-mllm-benchmark.git
cd posas-mllm-benchmark
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows PowerShell
pip install -r requirements.txt
```

---

## API keys

Copy the template and fill in the providers you plan to use:

```bash
cp .env.example .env
```

Then open `.env` and paste your keys:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AI...
OPENROUTER_API_KEY=sk-or-...
```

You do **not** need all four keys. Stage 3 will automatically skip any model
whose environment variable is not set and log a warning at startup.

---

## Adding your own images

1. Drop your scar photographs into `data/raw_images/`.
   Supported formats: JPG, JPEG, PNG, BMP, TIFF, WebP, HEIC, HEIF.
2. Run Stage 1 to register them:
   ```bash
   python pipeline.py --stage 1
   ```
   This writes `data/manifest.csv` with sequential IDs (`IMG_001`, `IMG_002` …).
3. (Optional) Open `data/manifest.csv` in Excel and fill in the metadata
   columns (`scar_type_label`, `scar_location`, `fitzpatrick_type`, `notes`).
   These are propagated through to the final export.

---

## Running the pipeline

```bash
# Register + preprocess images
python pipeline.py --stage 1,2

# Cost preview (dry run — will prompt before spending money)
python pipeline.py --stage 3

# Full API dispatch + validation + export (auto-confirms cost prompt)
python pipeline.py --stage 3,4,5 --yes

# Or run all five stages at once
python pipeline.py --stage all --yes

# Pilot run on the first 15 images only
python pipeline.py --stage 3,4,5 --pilot --yes

# Filter to a specific set of images
python pipeline.py --stage 3,4,5 --images S001.jpg,S002.jpg --yes
```

Stage 3 is **resumable** — if the run is interrupted, re-running with the same
command skips images that already have validated responses and only retries
failed ones.

---

## Outputs

After a full run, `outputs/` contains:

```
outputs/
├── raw_responses/         # Per-model, per-image raw JSON from each API
│   ├── chatgpt/IMG_001.json
│   ├── claude/IMG_001.json
│   └── ...
├── validated/             # Same shape, but parsed + schema-checked
│   └── ...
├── run_log.csv            # Every API call with tokens, cost, HTTP status
├── pipeline.log           # Timestamped structured log of the whole run
├── results_master.xlsx    # Final analysis-ready spreadsheet
└── results_master.csv     # Same as sheet 1 of the xlsx
```

`results_master.xlsx` has one row per image and, for each model, five item
scores + a total + a `valid_json` flag. Two blank rater columns (`h1_*`,
`h2_*`) are provided for you to paste in your human observer scores.

---

## The prompt

Every model receives the same zero-shot prompt (`prompts/posas_prompt.txt`).
Each model must return a single JSON object:

```json
{
  "observer_scores": {
    "vascularity": 5,
    "pigmentation": 3,
    "thickness": 4,
    "relief": 4,
    "surface_area": 2
  }
}
```

Any response that fails to parse or violates `config/posas_schema.json` is
flagged in Stage 4 and recorded in the audit log.

---

## Repository layout

```
posas-mllm-benchmark/
├── pipeline.py               # Orchestrator — run this
├── requirements.txt
├── .env.example
├── config/
│   ├── settings.yaml         # Model list, paths, retry policy, pricing
│   └── posas_schema.json     # JSON schema for a valid response
├── prompts/
│   └── posas_prompt.txt      # The single zero-shot prompt
├── src/
│   ├── stage1_ingest.py
│   ├── stage2_preprocess.py
│   ├── stage3_dispatch.py
│   ├── stage4_validate.py
│   ├── stage5_export.py
│   ├── adapters/             # One adapter per provider
│   │   ├── base_adapter.py
│   │   ├── openai_adapter.py
│   │   ├── anthropic_adapter.py
│   │   ├── gemini_adapter.py
│   │   └── openrouter_adapter.py
│   └── utils/
│       ├── logger.py
│       ├── image_utils.py
│       ├── cost_tracker.py
│       └── retry_handler.py
├── data/
│   ├── raw_images/           # ← Drop your images here
│   └── processed_images/     # ← Stage 2 writes here
└── outputs/                  # ← Everything else lands here
```

---

## Reproducing the published study

The exact model strings, pricing, and generation settings used in the paper
are pinned in `config/settings.yaml`. Model providers occasionally retire
versions — if a call fails with a 404 or "model not found" error, check the
provider's current model catalog and update the `model_string` field.

The paper reports results averaged across two independent inference runs.
To produce a second run, back up `outputs/` (or point `paths.raw_responses`
and `paths.validated` at a new directory in `settings.yaml`) before re-running
Stages 3–5. Stage 3 will skip existing responses, so it will only re-call
the API when the target output slot is empty.

---

## Adding a new model

1. Add an entry under `models:` in `config/settings.yaml`. Reuse an existing
   adapter if the provider is OpenAI-, Anthropic-, or Gemini-compatible, or
   route it through `openrouter_adapter` if OpenRouter hosts it.
2. If you need a brand-new adapter, subclass `BaseAdapter` in
   `src/adapters/base_adapter.py` and implement its six methods.
3. Register the class in `_ADAPTER_MAP` at the top of `src/stage3_dispatch.py`.

---

## Notes on content moderation

Some providers (notably Alibaba Qwen and Zhipu GLM through OpenRouter) apply
content-safety filters that occasionally refuse clinical scar images. These
failures are recorded in `outputs/validated/<model>/<image>.json` with a
descriptive error and counted separately in the Stage 4 summary. They are
**not** treated as pipeline bugs — they are provider-side refusals.

---

## Citing this software

If you use this pipeline in academic work, please cite the accompanying paper.
Citation details will be added on acceptance.

---

## License

MIT — see `LICENSE`.
