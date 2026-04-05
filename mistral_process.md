This document records the full experiment process for `mistralai/Mistral-7B-Instruct-v0.1`.

All outputs are isolated from the existing Qwen3 experiments:

- predictions and filtered sets go to `artifacts_mistral/`
- reports go to `reports_mistral/`
- MLP split datasets go to `dataset_mistral/`

Recommended model id:

```bash
MODEL_ID="mistralai/Mistral-7B-Instruct-v0.1"
```

Recommended device assignment:

- `INFERENCE_DEVICE`: the GPU used for `transformers` inference
- `HEADS_DEVICE`: the GPU used for `TransformerLens` head analysis
- `MLP_DEVICE`: the GPU used for `TransformerLens` MLP analysis
- `TOKEN_DEVICE`: the GPU used for `TransformerLens` token analysis

If you only use one GPU, replace all devices below with the same card.

Example:

```bash
INFERENCE_DEVICE="cuda:0"
HEADS_DEVICE="cuda:1"
MLP_DEVICE="cuda:1"
TOKEN_DEVICE="cuda:1"
```

You can run the entire Mistral pipeline with one command:

```bash
INFERENCE_DEVICE=cuda:0 \
HEADS_DEVICE=cuda:1 \
MLP_DEVICE=cuda:1 \
TOKEN_DEVICE=cuda:1 \
bash scripts/run_mistral_full.sh
```

If you prefer to run each stage manually, use the detailed commands below.

## 0. Generate the evaluation datasets

Facts-first dataset:

```bash
python -m src.data.generate_dataset \
  --output dataset/proplogic_mi.jsonl \
  --target_count 10000 \
  --one_hop_ratio 0.5 \
  --seed 42 \
  --prompt_order facts_first
```

Expr-first dataset:

```bash
python -m src.data.generate_dataset \
  --output dataset/proplogic_mi_expr_first.jsonl \
  --target_count 10000 \
  --one_hop_ratio 0.5 \
  --seed 42 \
  --prompt_order expr_first
```

## 1. Inference

Facts-first:

```bash
python -m src.eval.inference \
  --model_id mistralai/Mistral-7B-Instruct-v0.1 \
  --model_source huggingface \
  --input dataset/proplogic_mi.jsonl \
  --output artifacts_mistral/preds_facts_first.jsonl \
  --prompt_style symbolic \
  --max_new_tokens 1 \
  --mode nocot \
  --temperature 0.0 \
  --max_samples 0 \
  --dtype auto \
  --device cuda:0
```

Expr-first:

```bash
python -m src.eval.inference \
  --model_id mistralai/Mistral-7B-Instruct-v0.1 \
  --model_source huggingface \
  --input dataset/proplogic_mi_expr_first.jsonl \
  --output artifacts_mistral/preds_expr_first.jsonl \
  --prompt_style symbolic \
  --max_new_tokens 1 \
  --mode nocot \
  --temperature 0.0 \
  --max_samples 0 \
  --dtype auto \
  --device cuda:0
```

## 2. Filter dual-correct samples

Facts-first:

```bash
python -m src.eval.filtering \
  --input artifacts_mistral/preds_facts_first.jsonl \
  --output artifacts_mistral/filtered_dual_correct_facts_first.jsonl
```

Expr-first:

```bash
python -m src.eval.filtering \
  --input artifacts_mistral/preds_expr_first.jsonl \
  --output artifacts_mistral/filtered_dual_correct_expr_first.jsonl
```

## 3. Compute metrics

Facts-first:

```bash
python -m src.eval.metrics \
  --input artifacts_mistral/preds_facts_first.jsonl \
  --output reports_mistral/metrics_facts_first.json
```

Expr-first:

```bash
python -m src.eval.metrics \
  --input artifacts_mistral/preds_expr_first.jsonl \
  --output reports_mistral/metrics_expr_first.json
```

## 4. MLP region patching

Facts-first:

```bash
bash scripts/MLP_region_patching.sh \
  --model_id "mistralai/Mistral-7B-Instruct-v0.1" \
  --model_source "huggingface" \
  --prompt_style "symbolic" \
  --device "cuda:1" \
  --max_samples 1000 \
  --progress_every 100 \
  --split "all" \
  --source_input "artifacts_mistral/filtered_dual_correct_facts_first.jsonl" \
  --one_hop_input "dataset_mistral/mlp_one_hop_facts_first.jsonl" \
  --two_hop_input "dataset_mistral/mlp_two_hop_facts_first.jsonl" \
  --split_summary "reports_mistral/mlp_analysis_facts_first/dual_correct_split.json" \
  --output_root "reports_mistral/mlp_analysis_facts_first" \
  --auto_split 1 \
  --make_plots 1 \
  --plot_output_dir "reports_mistral/mlp_analysis_facts_first/comparison"
```

Expr-first:

```bash
bash scripts/MLP_region_patching.sh \
  --model_id "mistralai/Mistral-7B-Instruct-v0.1" \
  --model_source "huggingface" \
  --prompt_style "symbolic" \
  --device "cuda:0" \
  --max_samples 1000 \
  --progress_every 100 \
  --split "all" \
  --source_input "artifacts_mistral/filtered_dual_correct_expr_first.jsonl" \
  --one_hop_input "dataset_mistral/mlp_one_hop_expr_first.jsonl" \
  --two_hop_input "dataset_mistral/mlp_two_hop_expr_first.jsonl" \
  --split_summary "reports_mistral/mlp_analysis_expr_first/dual_correct_split.json" \
  --output_root "reports_mistral/mlp_analysis_expr_first" \
  --auto_split 1 \
  --make_plots 1 \
  --plot_output_dir "reports_mistral/mlp_analysis_expr_first/comparison"
```

## 5. Token analysis

For Mistral-7B, use the current stage split:

- `early_end=12`
- `middle_end=22`

Facts-first raw patching:

```bash
python3 -m src.token_analysis.activation_patching_dataset \
  --model_id mistralai/Mistral-7B-Instruct-v0.1 \
  --model_source huggingface \
  --input artifacts_mistral/filtered_dual_correct_facts_first.jsonl \
  --output_dir reports_mistral/token_analysis_facts_first_raw \
  --prompt_style symbolic \
  --hop all \
  --max_samples 1000 \
  --require_dual_correct \
  --strict_length_match \
  --early_end 12 \
  --middle_end 22 \
  --device cuda:1 \
  --progress_every 10 \
  --save_plots
```

Facts-first refined analysis:

```bash
python3 -m src.token_analysis.refined_token_analysis \
  --input_pkl reports_mistral/token_analysis_facts_first_raw/patching_results.pkl \
  --output_dir reports_mistral/token_analysis_facts_first_refined \
  --title "Mistral-7B-Instruct-v0.1 Facts-first All-hop Refined Token Analysis" \
  --early_end 12 \
  --middle_end 22 \
  --n_layers 0 \
  --include-derived-assignment \
  --save-plots \
  --save-csv
```

Expr-first raw patching:

```bash
python3 -m src.token_analysis.activation_patching_dataset \
  --model_id mistralai/Mistral-7B-Instruct-v0.1 \
  --model_source huggingface \
  --input artifacts_mistral/filtered_dual_correct_expr_first.jsonl \
  --output_dir reports_mistral/token_analysis_expr_first_raw \
  --prompt_style symbolic \
  --hop all \
  --max_samples 1000 \
  --require_dual_correct \
  --strict_length_match \
  --early_end 12 \
  --middle_end 22 \
  --device cuda:1 \
  --progress_every 10 \
  --save_plots
```

Expr-first refined analysis:

```bash
python3 -m src.token_analysis.refined_token_analysis \
  --input_pkl reports_mistral/token_analysis_expr_first_raw/patching_results.pkl \
  --output_dir reports_mistral/token_analysis_expr_first_refined \
  --title "Mistral-7B-Instruct-v0.1 Expr-first All-hop Refined Token Analysis" \
  --early_end 12 \
  --middle_end 22 \
  --n_layers 0 \
  --include-derived-assignment \
  --save-plots \
  --save-csv
```

## 6. Heads analysis

Facts-first:

```bash
python3 -m src.heads_analysis.run_all \
  --model_id mistralai/Mistral-7B-Instruct-v0.1 \
  --model_source huggingface \
  --output_dir reports_mistral/heads_analysis_facts_first \
  --input artifacts_mistral/filtered_dual_correct_facts_first.jsonl \
  --hop all \
  --prompt_order facts_first \
  --prompt_style symbolic \
  --token_scope all_tokens \
  --impact_samples 2000 \
  --probe_samples 256 \
  --classify_samples 2000 \
  --validation_samples 1000 \
  --validation_accuracy_samples 500 \
  --top_n 512 \
  --top_m_per_layer 4 \
  --candidate_pool_mult 4 \
  --quantile_keep 0.6 \
  --late_layer_frac 0.0 \
  --score_mode zscore \
  --k_values 1,2,4,8,16,32,64 \
  --random_trials_min 6 \
  --random_trials_max 20 \
  --random_sem_target 0.01 \
  --eval_batch_size 16 \
  --seed 42 \
  --device cuda:1 \
  --steps 1,2,3
```

Expr-first:

```bash
python3 -m src.heads_analysis.run_all \
  --model_id mistralai/Mistral-7B-Instruct-v0.1 \
  --model_source huggingface \
  --output_dir reports_mistral/heads_analysis_expr_first \
  --input artifacts_mistral/filtered_dual_correct_expr_first.jsonl \
  --hop all \
  --prompt_order expr_first \
  --prompt_style symbolic \
  --token_scope all_tokens \
  --impact_samples 2000 \
  --probe_samples 256 \
  --classify_samples 2000 \
  --validation_samples 1000 \
  --validation_accuracy_samples 500 \
  --top_n 512 \
  --top_m_per_layer 4 \
  --candidate_pool_mult 4 \
  --quantile_keep 0.6 \
  --late_layer_frac 0.0 \
  --score_mode zscore \
  --k_values 1,2,4,8,16,32,64 \
  --random_trials_min 6 \
  --random_trials_max 20 \
  --random_sem_target 0.01 \
  --eval_batch_size 16 \
  --seed 42 \
  --device cuda:1 \
  --steps 1,2,3
```

## 7. Stage-wise rerun with the new wrapper

If you want isolated outputs but prefer the wrapper script, these are the most useful variants.

Run the full Mistral pipeline:

```bash
INFERENCE_DEVICE=cuda:0 \
HEADS_DEVICE=cuda:1 \
MLP_DEVICE=cuda:1 \
TOKEN_DEVICE=cuda:1 \
bash scripts/run_mistral_full.sh
```

Run only inference, filtering, and metrics:

```bash
RUN_HEADS=0 \
RUN_MLP=0 \
RUN_TOKEN=0 \
INFERENCE_DEVICE=cuda:0 \
bash scripts/run_mistral_full.sh
```

Run only the heavy `TransformerLens` stages after inference is already done:

```bash
RUN_INFERENCE=0 \
RUN_FILTERING=0 \
RUN_METRICS=0 \
HEADS_DEVICE=cuda:1 \
MLP_DEVICE=cuda:1 \
TOKEN_DEVICE=cuda:1 \
bash scripts/run_mistral_full.sh
```

## 8. Clear logs

```bash
bash scripts/clear_logs.sh
```
