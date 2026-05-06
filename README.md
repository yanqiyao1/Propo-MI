# PropLogic-MI

Experimental code for reproducing the mechanistic-interpretability experiments on propositional logical reasoning in large language models.

This repository contains the data generator, model evaluation pipeline, activation-patching analyses, attention-head taxonomy, and plotting utilities used for the PropLogic-MI experiments. It intentionally excludes paper sources, compiled PDFs, generated datasets, model outputs, reports, and figures from version control; those files are produced locally by the commands below.

## What This Code Reproduces

PropLogic-MI studies how instruction-tuned LLMs solve controlled propositional-logic prompts without chain-of-thought generation. The experiments are organized around four mechanisms:

1. **Staged Computation**: region-level MLP, attention, and joint attention+MLP interventions show that facts, expressions, and query integration occupy different layer bands.
2. **Information Transmission**: token-wise residual-stream patching shows where causal information concentrates across prompt positions and layers.
3. **Fact Retrospection**: refined token analysis shows that fact-value tokens remain causally relevant beyond the earliest layers.
4. **Specialized Attention Heads**: head discovery, taxonomy, and validation recover Splitting, Transmission, and Fact-Retrieval head roles.

The main supported models are:

- `Qwen/Qwen3-8B`
- `Qwen/Qwen3-14B`
- `meta-llama/Llama-3.1-8B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.1`

Both prompt orders used in the paper are supported:

- `facts_first`: facts precede the queried expression.
- `expr_first`: the queried expression is moved before the facts as a prompt-order control.

## Repository Layout

```text
src/
  data/                  PropLogic-MI dataset generation and hop splitting
  eval/                  model inference, dual-correct filtering, metrics
  mlp_analysis/          region-level MLP ablation and plotting
  attn_analysis/         region-level attention ablation and plotting
  attn_mlp_analysis/     joint attention+MLP region ablation
  token_analysis/        token-wise residual patching and refined token analysis
  heads_analysis/        attention-head discovery, taxonomy, validation, plots
  mech/                  lower-level patching utilities
scripts/
  dataset_create.sh      create facts-first and expr-first datasets
  qwen3_*.sh             Qwen3 reproduction wrappers
  llama3_*.sh            Llama-3.1 wrappers
  mistral_*.sh           Mistral wrappers
  model_*.sh             generic per-model pipelines
  replot_*.sh            regenerate figures from saved reports
```

Generated files are written under `dataset*/`, `artifacts*/`, `reports*/`, `figures*/`, and `logs/`. These directories are ignored by Git.

## Environment

The experiments require CUDA-capable GPUs. The paper-scale runs were designed for high-memory GPUs; smaller smoke tests can be run by lowering `MAX_SAMPLES`, `TOKEN_MAX_SAMPLES`, `MLP_MAX_SAMPLES`, and related variables.

```bash
conda create -n proplogic-mi python=3.10 -y
conda activate proplogic-mi
pip install -r requirements.txt
```

Model loading supports both Hugging Face and ModelScope:

```bash
export MODEL_SOURCE=huggingface   # or modelscope
```

For gated Hugging Face models such as Llama-3.1, make sure your local environment is already authenticated.

## Reproduction Guide

Run all commands from the repository root.

### 1. Build PropLogic-MI Datasets

This generates the two 20k-row prompt-order variants used throughout the experiments.

```bash
bash scripts/dataset_create.sh
```

Outputs:

```text
dataset/proplogic_mi.jsonl
dataset/proplogic_mi_expr_first.jsonl
```

Low-level equivalent:

```bash
python -m src.data.generate_dataset \
  --output dataset/proplogic_mi.jsonl \
  --target_count 20000 \
  --one_hop_ratio 0.5 \
  --seed 42 \
  --prompt_order facts_first

python -m src.data.generate_dataset \
  --output dataset/proplogic_mi_expr_first.jsonl \
  --target_count 20000 \
  --one_hop_ratio 0.5 \
  --seed 42 \
  --prompt_order expr_first
```

### 2. Behavioral Evaluation and Dual-Correct Filtering

The mechanistic experiments use only examples where the model answers both the clean and corrupted prompts correctly.

For Qwen3-8B and Qwen3-14B:

```bash
MODEL_SOURCE=huggingface \
INFERENCE_DEVICE=cuda:0 \
bash scripts/qwen3_inference.sh
```

Outputs include:

```text
artifacts/preds_Qwen3-8b.jsonl
artifacts/preds_Qwen3-14b.jsonl
artifacts/preds_Qwen3-8b_expr_first.jsonl
artifacts/preds_Qwen3-14b_expr_first.jsonl
artifacts/filtered_dual_correct_Qwen3-8b.jsonl
artifacts/filtered_dual_correct_Qwen3-14b.jsonl
reports/metrics_Qwen3-8b.json
reports/metrics_Qwen3-14b.json
```

Generic single-model pipeline:

```bash
MODEL_ID=Qwen/Qwen3-8B \
MODEL_SOURCE=huggingface \
ARTIFACT_DIR=artifacts_qwen3_8b \
REPORT_DIR=reports_qwen3_8b \
INFERENCE_DEVICE=cuda:0 \
bash scripts/model_inference_suite.sh
```

### 3. Experiment 1: Staged Computation

Run the three region-level branches in the order used by the paper: MLP, attention, then joint attention+MLP.

#### 3.1 MLP Region Ablation

```bash
MODEL_ID=Qwen/Qwen3-8B \
MODEL_SOURCE=huggingface \
ARTIFACT_DIR=artifacts_qwen3_8b \
REPORT_DIR=reports_qwen3_8b \
DATASET_DIR=dataset_qwen3_8b \
MLP_DEVICE=cuda:0 \
MLP_MAX_SAMPLES=1000 \
bash scripts/model_mlp_patching.sh
```

For the historical Qwen paper-output paths, use:

```bash
REGION_DEVICE=cuda:0 \
REGION_MAX_SAMPLES=1000 \
MODEL_SOURCE=huggingface \
bash scripts/qwen3_mlp_patching.sh
```

#### 3.2 Attention Region Ablation

```bash
MODEL_ID=Qwen/Qwen3-8B \
MODEL_SOURCE=huggingface \
ARTIFACT_DIR=artifacts_qwen3_8b \
REPORT_DIR=reports_qwen3_8b \
DATASET_DIR=dataset_qwen3_8b \
ATTN_DEVICE=cuda:0 \
ATTN_MAX_SAMPLES=1000 \
bash scripts/model_attention_patching.sh
```

#### 3.3 Joint Attention+MLP Region Ablation

```bash
MODEL_ID=Qwen/Qwen3-8B \
MODEL_SOURCE=huggingface \
ARTIFACT_DIR=artifacts_qwen3_8b \
REPORT_DIR=reports_qwen3_8b \
DATASET_DIR=dataset_qwen3_8b \
ATTN_MLP_DEVICE=cuda:0 \
ATTN_MLP_MAX_SAMPLES=1000 \
bash scripts/model_attention_mlp_patching.sh
```

Each branch automatically splits the dual-correct pool into one-hop and two-hop subsets and writes region score plots plus band-comparison panels under `reports*/`.

### 4. Experiment 2: Information Transmission

Token-wise residual patching saves the raw layer-by-token patching tensor and then runs simple and refined token-category analyses.

```bash
MODEL_ID=Qwen/Qwen3-8B \
MODEL_SOURCE=huggingface \
ARTIFACT_DIR=artifacts_qwen3_8b \
REPORT_DIR=reports_qwen3_8b \
TOKEN_DEVICE=cuda:0 \
TOKEN_MAX_SAMPLES=1000 \
N_LAYERS=36 \
MODEL_LABEL=Qwen3-8B \
bash scripts/model_token_analysis.sh
```

Important outputs:

```text
reports_qwen3_8b/token_analysis_facts_first_raw/patching_results.pkl
reports_qwen3_8b/token_analysis_facts_first_refined/refined_stats_sum.json
reports_qwen3_8b/token_analysis_facts_first_refined/refined_stats_mean.json
reports_qwen3_8b/token_analysis_expr_first_refined/refined_by_stage_mean.png
```

### 5. Experiment 3: Fact Retrospection

There is no separate expensive pipeline for this claim. It is read from the refined token-analysis outputs produced in Step 4. In particular, inspect the `facts_value` category across early, middle, and late bands:

```bash
python -m src.token_analysis.plot_refined_analysis \
  --sum_stats_json reports_qwen3_8b/token_analysis_facts_first_refined/refined_stats_sum.json \
  --mean_stats_json reports_qwen3_8b/token_analysis_facts_first_refined/refined_stats_mean.json \
  --output_dir reports_qwen3_8b/token_analysis_facts_first_refined
```

The same command applies to the `expr_first` refined directory.

### 6. Experiment 4: Specialized Attention Heads

This pipeline runs head discovery, taxonomy plotting, and held-out validation.

```bash
MODEL_ID=Qwen/Qwen3-8B \
MODEL_SOURCE=huggingface \
ARTIFACT_DIR=artifacts_qwen3_8b \
REPORT_DIR=reports_qwen3_8b \
HEADS_DEVICE=cuda:0 \
IMPACT_SAMPLES=2000 \
CLASSIFY_SAMPLES=2000 \
VALIDATION_SAMPLES=1000 \
bash scripts/model_heads_analysis.sh
```

Important outputs:

```text
reports_qwen3_8b/heads_analysis_facts_first/classify/top_heads_pattern_labels.csv
reports_qwen3_8b/heads_analysis_facts_first/taxonomy/head_taxonomy_counts.csv
reports_qwen3_8b/heads_analysis_facts_first/validation/pd_curve_metrics.csv
reports_qwen3_8b/heads_analysis_facts_first/validation/plots/
```

For the original Qwen3 paper-output paths:

```bash
bash scripts/qwen3_head_analysis.sh
```

### 7. Prompt-Order Control

All wrapper scripts above run both `facts_first` and `expr_first` when both filtered input files exist. To run only one prompt order, call the low-level module directly, for example:

```bash
python -m src.heads_analysis.run_all \
  --model_id Qwen/Qwen3-8B \
  --model_source huggingface \
  --input artifacts/filtered_dual_correct_Qwen3-8b_expr_first.jsonl \
  --output_dir reports/heads_analysis_qwen3_8b_expr_first \
  --hop all \
  --prompt_order expr_first \
  --prompt_style symbolic \
  --token_scope all_tokens \
  --device cuda:0 \
  --steps 1,2,3
```

### 8. Cross-Model Runs

Llama-3.1-8B-Instruct:

```bash
MODEL_SOURCE=huggingface \
INFERENCE_DEVICE=cuda:0 \
HEADS_DEVICE=cuda:0 \
MLP_DEVICE=cuda:0 \
TOKEN_DEVICE=cuda:0 \
bash scripts/run_llama3_full.sh
```

Mistral-7B-Instruct:

```bash
MODEL_SOURCE=huggingface \
INFERENCE_DEVICE=cuda:0 \
HEADS_DEVICE=cuda:0 \
MLP_DEVICE=cuda:0 \
TOKEN_DEVICE=cuda:0 \
bash scripts/run_mistral_full.sh
```

These scripts use a local checkpoint under `/media/snail-ssd/models/...` if present; otherwise they fall back to the public model IDs.

### 9. Regenerate Figures from Saved Reports

After the experiment outputs exist:

```bash
bash scripts/replot_all_figures.sh
```

This redraws region, token, and head figures and synchronizes PNG/PDF outputs into `figures_experiments/`.

## Useful Smoke Tests

For quick checks, lower sample counts:

```bash
python -m src.data.generate_dataset \
  --output dataset/smoke.jsonl \
  --target_count 200 \
  --seed 42 \
  --prompt_order facts_first

python -m src.eval.inference \
  --model_id Qwen/Qwen3-8B \
  --model_source huggingface \
  --input dataset/smoke.jsonl \
  --output artifacts/smoke_preds.jsonl \
  --prompt_style symbolic \
  --max_new_tokens 1 \
  --mode nocot \
  --temperature 0.0 \
  --max_samples 20 \
  --device cuda:0
```

## Output Hygiene

The repository is configured to track source code and reproduction scripts only. These paths are ignored by design:

```text
dataset*/       generated datasets and split files
artifacts*/     inference outputs and filtered dual-correct pools
reports*/       CSV/JSON/PKL analysis outputs and plots
figures*/       rendered figures
logs/           run logs
*.tex, *.pdf    paper source/build products
.texenv/        local TeX environment
.tectonic-*/    TeX build/cache directories
```

If you need to clear run logs:

```bash
bash scripts/clear_logs.sh
```

## Notes

- Most expensive analyses rely on TransformerLens hooks and should be run with `torch.no_grad()` through the provided modules.
- The default scripts use symbolic prompts and no chain-of-thought generation.
- `MAX_SAMPLES=0` usually means "use all available examples"; for paper-scale reproduction, keep the larger defaults in the wrapper scripts.
- All generated files can be safely deleted and regenerated from the tracked code.
