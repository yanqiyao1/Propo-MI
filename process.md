# Propologic-MI 实验复现说明

下面记录当前项目的推荐复现实验流程。

重要更新：现在分析实验统一支持“结果计算”和“画图”分离。
也就是说：

1. 先跑实验，保存中间结果文件（csv/json/pkl 等）
2. 再根据这些结果文件单独画图

这样更方便断点续跑、复用旧结果、重新调整画图样式，而且不会影响之前的 Qwen3 实验目录结构。

## 0. 生成数据集

Facts-first:

```bash
python -m src.data.generate_dataset \
  --output dataset/proplogic_mi.jsonl \
  --target_count 20000 \
  --one_hop_ratio 0.5 \
  --seed 42 \
  --prompt_order facts_first
```

Expr-first:

```bash
python -m src.data.generate_dataset \
  --output dataset/proplogic_mi_expr_first.jsonl \
  --target_count 20000 \
  --one_hop_ratio 0.5 \
  --seed 42 \
  --prompt_order expr_first
```

## 1. 推理

Qwen3-8B / facts-first:

```bash
python -m src.eval.inference \
  --model_id Qwen/Qwen3-8B \
  --model_source huggingface \
  --input dataset/proplogic_mi.jsonl \
  --output artifacts/preds_Qwen3-8b.jsonl \
  --prompt_style symbolic \
  --max_new_tokens 1 \
  --mode nocot \
  --temperature 0.0 \
  --max_samples 0 \
  --dtype auto \
  --device cuda:0
```

Qwen3-14B / facts-first:

```bash
python -m src.eval.inference \
  --model_id Qwen/Qwen3-14B \
  --model_source huggingface \
  --input dataset/proplogic_mi.jsonl \
  --output artifacts/preds_Qwen3-14b.jsonl \
  --prompt_style symbolic \
  --max_new_tokens 1 \
  --mode nocot \
  --temperature 0.0 \
  --max_samples 0 \
  --dtype auto \
  --device cuda:0
```

Qwen3-8B / expr-first:

```bash
python -m src.eval.inference \
  --model_id Qwen/Qwen3-8B \
  --model_source huggingface \
  --input dataset/proplogic_mi_expr_first.jsonl \
  --output artifacts/preds_Qwen3-8b_expr_first.jsonl \
  --prompt_style symbolic \
  --max_new_tokens 1 \
  --mode nocot \
  --temperature 0.0 \
  --max_samples 0 \
  --dtype auto \
  --device cuda:0
```

Qwen3-14B / expr-first:

```bash
python -m src.eval.inference \
  --model_id Qwen/Qwen3-14B \
  --model_source huggingface \
  --input dataset/proplogic_mi_expr_first.jsonl \
  --output artifacts/preds_Qwen3-14b_expr_first.jsonl \
  --prompt_style symbolic \
  --max_new_tokens 1 \
  --mode nocot \
  --temperature 0.0 \
  --max_samples 0 \
  --dtype auto \
  --device cuda:0
```

## 2. 过滤 dual-correct 样本

Facts-first:

```bash
python -m src.eval.filtering \
  --input artifacts/preds_Qwen3-8b.jsonl \
  --output artifacts/filtered_dual_correct_Qwen3-8b.jsonl
```

```bash
python -m src.eval.filtering \
  --input artifacts/preds_Qwen3-14b.jsonl \
  --output artifacts/filtered_dual_correct_Qwen3-14b.jsonl
```

Expr-first:

```bash
python -m src.eval.filtering \
  --input artifacts/preds_Qwen3-8b_expr_first.jsonl \
  --output artifacts/filtered_dual_correct_Qwen3-8b_expr_first.jsonl
```

```bash
python -m src.eval.filtering \
  --input artifacts/preds_Qwen3-14b_expr_first.jsonl \
  --output artifacts/filtered_dual_correct_Qwen3-14b_expr_first.jsonl
```

## 3. 计算基础指标

```bash
python -m src.eval.metrics \
  --input artifacts/preds_Qwen3-8b.jsonl \
  --output reports/metrics_Qwen3-8b.json
```

```bash
python -m src.eval.metrics \
  --input artifacts/preds_Qwen3-14b.jsonl \
  --output reports/metrics_Qwen3-14b.json
```

```bash
python -m src.eval.metrics \
  --input artifacts/preds_Qwen3-8b_expr_first.jsonl \
  --output reports/metrics_Qwen3-8b_expr_first.json
```

```bash
python -m src.eval.metrics \
  --input artifacts/preds_Qwen3-14b_expr_first.jsonl \
  --output reports/metrics_Qwen3-14b_expr_first.json
```

## 4. MLP region patching

这里已经改成两阶段：

- 第一阶段：`scripts/MLP_region_patching.sh` 只生成结果文件，不画图
- 第二阶段：根据保存好的 `csv/json` 单独画图

### 4.1 先生成结果

Qwen3-8B / facts-first:

```bash
bash scripts/MLP_region_patching.sh \
  --model_id "Qwen/Qwen3-8B" \
  --model_source "huggingface" \
  --prompt_style "symbolic" \
  --device "cuda:0" \
  --max_samples 3000 \
  --progress_every 100 \
  --split "all" \
  --source_input "artifacts/filtered_dual_correct_Qwen3-8b.jsonl" \
  --one_hop_input "dataset/mlp_8b_one_hop.jsonl" \
  --two_hop_input "dataset/mlp_8b_two_hop.jsonl" \
  --split_summary "reports/mlp_analysis/8b_dual_correct_split.json" \
  --output_root "reports/mlp_analysis" \
  --auto_split 1 \
  --make_plots 0 \
  --plot_output_dir "reports/mlp_analysis/comparison"
```

Qwen3-14B / facts-first:

```bash
bash scripts/MLP_region_patching.sh \
  --model_id "Qwen/Qwen3-14B" \
  --model_source "huggingface" \
  --prompt_style "symbolic" \
  --device "cuda:1" \
  --max_samples 3000 \
  --progress_every 100 \
  --split "all" \
  --source_input "artifacts/filtered_dual_correct_Qwen3-14b.jsonl" \
  --one_hop_input "dataset/mlp_14b_one_hop.jsonl" \
  --two_hop_input "dataset/mlp_14b_two_hop.jsonl" \
  --split_summary "reports/mlp_analysis/14b_dual_correct_split.json" \
  --output_root "reports/mlp_analysis" \
  --auto_split 1 \
  --make_plots 0 \
  --plot_output_dir "reports/mlp_analysis/comparison_14B"
```

Qwen3-8B / expr-first:

```bash
bash scripts/MLP_region_patching.sh \
  --model_id "Qwen/Qwen3-8B" \
  --model_source "huggingface" \
  --prompt_style "symbolic" \
  --device "cuda:0" \
  --max_samples 3000 \
  --progress_every 100 \
  --split "all" \
  --source_input "artifacts/filtered_dual_correct_Qwen3-8b_expr_first.jsonl" \
  --one_hop_input "dataset/mlp_8b_one_hop_expr_first.jsonl" \
  --two_hop_input "dataset/mlp_8b_two_hop_expr_first.jsonl" \
  --split_summary "reports/mlp_analysis/8b_dual_correct_split_expr_first.json" \
  --output_root "reports/mlp_analysis_expr_first" \
  --auto_split 1 \
  --make_plots 0 \
  --plot_output_dir "reports/mlp_analysis_expr_first/comparison_expr_first"
```

Qwen3-14B / expr-first:

```bash
bash scripts/MLP_region_patching.sh \
  --model_id "Qwen/Qwen3-14B" \
  --model_source "huggingface" \
  --prompt_style "symbolic" \
  --device "cuda:1" \
  --max_samples 3000 \
  --progress_every 100 \
  --split "all" \
  --source_input "artifacts/filtered_dual_correct_Qwen3-14b_expr_first.jsonl" \
  --one_hop_input "dataset/mlp_14b_one_hop_expr_first.jsonl" \
  --two_hop_input "dataset/mlp_14b_two_hop_expr_first.jsonl" \
  --split_summary "reports/mlp_analysis/14b_dual_correct_split_expr_first.json" \
  --output_root "reports/mlp_analysis_expr_first" \
  --auto_split 1 \
  --make_plots 0 \
  --plot_output_dir "reports/mlp_analysis_expr_first/comparison_expr_first_14B"
```

### 4.2 再单独画图

单个 region 的柱状图示例（把路径替换成你想画的 region 目录即可）：

```bash
python -m src.mlp_analysis.plot_region_scores \
  --scores_csv reports/mlp_analysis/mlp_8b_one_hop/facts_region/mlp_facts_region_scores.csv \
  --output_png reports/mlp_analysis/mlp_8b_one_hop/facts_region/mlp_facts_region_scores.png
```

MLP band comparison 图示例：

```bash
python -m src.mlp_analysis.plot_band_metrics \
  --one_hop_dir reports/mlp_analysis/mlp_8b_one_hop \
  --two_hop_dir reports/mlp_analysis/mlp_8b_two_hop \
  --output_dir reports/mlp_analysis/comparison
```

如果你想直接一条命令同时“算结果 + 画图”，仍然可以把 `--make_plots 0` 改回 `--make_plots 1`。

## 5. Token analysis

这里也改成两阶段：

- `src.token_analysis.activation_patching_dataset`：生成 raw patching 结果
- `src.token_analysis.plot_simple_analysis`：根据 `patching_results.pkl` 单独画 simple 图
- `src.token_analysis.refined_token_analysis`：生成 refined 统计结果
- `src.token_analysis.plot_refined_analysis`：根据 refined json 单独画 refined 图

### 5.1 生成 raw patching 结果

Qwen3-14B / facts-first:

```bash
python3 -m src.token_analysis.activation_patching_dataset \
  --model_id Qwen/Qwen3-14B \
  --model_source huggingface \
  --input artifacts/filtered_dual_correct_Qwen3-14b.jsonl \
  --output_dir reports/token_analysis/14b_facts_first_all_hop_raw \
  --prompt_style symbolic \
  --hop all \
  --max_samples 1000 \
  --require_dual_correct \
  --strict_length_match \
  --early_end 14 \
  --middle_end 24 \
  --device cuda:0 \
  --progress_every 10 \
  --no-save_plots
```

Qwen3-14B / expr-first:

```bash
python3 -m src.token_analysis.activation_patching_dataset \
  --model_id Qwen/Qwen3-14B \
  --model_source huggingface \
  --input artifacts/filtered_dual_correct_Qwen3-14b_expr_first.jsonl \
  --output_dir reports/token_analysis/14b_expr_first_all_hop_raw \
  --prompt_style symbolic \
  --hop all \
  --max_samples 1000 \
  --require_dual_correct \
  --strict_length_match \
  --early_end 14 \
  --middle_end 24 \
  --device cuda:0 \
  --progress_every 10 \
  --no-save_plots
```

Qwen3-8B / facts-first:

```bash
python3 -m src.token_analysis.activation_patching_dataset \
  --model_id Qwen/Qwen3-8B \
  --model_source huggingface \
  --input artifacts/filtered_dual_correct_Qwen3-8b.jsonl \
  --output_dir reports/token_analysis/8b_facts_first_all_hop_raw \
  --prompt_style symbolic \
  --hop all \
  --max_samples 1000 \
  --require_dual_correct \
  --strict_length_match \
  --early_end 14 \
  --middle_end 24 \
  --device cuda:0 \
  --progress_every 10 \
  --no-save_plots
```

Qwen3-8B / expr-first:

```bash
python3 -m src.token_analysis.activation_patching_dataset \
  --model_id Qwen/Qwen3-8B \
  --model_source huggingface \
  --input artifacts/filtered_dual_correct_Qwen3-8b_expr_first.jsonl \
  --output_dir reports/token_analysis/8b_expr_first_all_hop_raw \
  --prompt_style symbolic \
  --hop all \
  --max_samples 1000 \
  --require_dual_correct \
  --strict_length_match \
  --early_end 14 \
  --middle_end 24 \
  --device cuda:0 \
  --progress_every 10 \
  --no-save_plots
```

### 5.2 根据 raw 结果单独画 simple 图

```bash
python3 -m src.token_analysis.plot_simple_analysis \
  --input_pkl reports/token_analysis/8b_facts_first_all_hop_raw/patching_results.pkl \
  --summary_json reports/token_analysis/8b_facts_first_all_hop_raw/summary.json \
  --output_dir reports/token_analysis/8b_facts_first_all_hop_raw
```

### 5.3 生成 refined 统计结果

Qwen3-14B / facts-first:

```bash
python3 -m src.token_analysis.refined_token_analysis \
  --input_pkl reports/token_analysis/14b_facts_first_all_hop_raw/patching_results.pkl \
  --output_dir reports/token_analysis/14b_facts_first_all_hop_refined \
  --title "Qwen3-14B Facts-first All-hop Refined Token Analysis" \
  --early_end 14 \
  --middle_end 24 \
  --n_layers 40 \
  --include-derived-assignment \
  --no-save-plots \
  --save-csv
```

Qwen3-14B / expr-first:

```bash
python3 -m src.token_analysis.refined_token_analysis \
  --input_pkl reports/token_analysis/14b_expr_first_all_hop_raw/patching_results.pkl \
  --output_dir reports/token_analysis/14b_expr_first_all_hop_refined \
  --title "Qwen3-14B Expr-first All-hop Refined Token Analysis" \
  --early_end 14 \
  --middle_end 24 \
  --n_layers 40 \
  --include-derived-assignment \
  --no-save-plots \
  --save-csv
```

Qwen3-8B / facts-first:

```bash
python3 -m src.token_analysis.refined_token_analysis \
  --input_pkl reports/token_analysis/8b_facts_first_all_hop_raw/patching_results.pkl \
  --output_dir reports/token_analysis/8b_facts_first_all_hop_refined \
  --title "Qwen3-8B Facts-first All-hop Refined Token Analysis" \
  --early_end 14 \
  --middle_end 24 \
  --n_layers 36 \
  --include-derived-assignment \
  --no-save-plots \
  --save-csv
```

Qwen3-8B / expr-first:

```bash
python3 -m src.token_analysis.refined_token_analysis \
  --input_pkl reports/token_analysis/8b_expr_first_all_hop_raw/patching_results.pkl \
  --output_dir reports/token_analysis/8b_expr_first_all_hop_refined \
  --title "Qwen3-8B Expr-first All-hop Refined Token Analysis" \
  --early_end 14 \
  --middle_end 24 \
  --n_layers 36 \
  --include-derived-assignment \
  --no-save-plots \
  --save-csv
```

### 5.4 根据 refined 结果单独画图

```bash
python3 -m src.token_analysis.plot_refined_analysis \
  --sum_stats_json reports/token_analysis/8b_facts_first_all_hop_refined/refined_stats_sum.json \
  --mean_stats_json reports/token_analysis/8b_facts_first_all_hop_refined/refined_stats_mean.json \
  --output_dir reports/token_analysis/8b_facts_first_all_hop_refined
```

## 6. Heads analysis

这里也改成两阶段：

- `src.heads_analysis.run_all`：只生成 step1/2/3 的 csv/json/md 等结果
- 单独使用 `plot_step1_heads` / `plot_step2_taxonomy` / `plot_step3_curves` 来画图

### 6.1 先生成结果

下面四条命令都建议加 `--no-save-plots`。
如果你使用别的 GPU，把 `--device` 改掉即可。

Qwen3-8B / facts-first:

```bash
python3 -m src.heads_analysis.run_all \
  --model_id Qwen/Qwen3-8B \
  --model_source huggingface \
  --output_dir reports/heads_analysis_qwen3_8b_facts_first \
  --input artifacts/filtered_dual_correct_Qwen3-8b.jsonl \
  --hop all \
  --prompt_order facts_first \
  --prompt_style symbolic \
  --token_scope all_tokens \
  --impact_samples 2000 \
  --probe_samples 256 \
  --classify_samples 2000 \
  --validation_samples 1000 \
  --validation_accuracy_samples 500 \
  --top_n 64 \
  --top_m_per_layer 4 \
  --candidate_pool_mult 4 \
  --quantile_keep 0.6 \
  --late_layer_frac 0.0 \
  --score_mode zscore \
  --k_values 1,2,4,8,16,32,64 \
  --random_trials_min 6 \
  --random_trials_max 20 \
  --random_sem_target 0.01 \
  --eval_batch_size 8 \
  --seed 42 \
  --device cuda:1 \
  --steps 1,2,3 \
  --no-save-plots
```

Qwen3-8B / expr-first:

```bash
python3 -m src.heads_analysis.run_all \
  --model_id Qwen/Qwen3-8B \
  --model_source huggingface \
  --output_dir reports/heads_analysis_qwen3_8b_expr_first \
  --input artifacts/filtered_dual_correct_Qwen3-8b_expr_first.jsonl \
  --hop all \
  --prompt_order expr_first \
  --prompt_style symbolic \
  --token_scope all_tokens \
  --impact_samples 2000 \
  --probe_samples 256 \
  --classify_samples 2000 \
  --validation_samples 1000 \
  --validation_accuracy_samples 500 \
  --top_n 64 \
  --top_m_per_layer 4 \
  --candidate_pool_mult 4 \
  --quantile_keep 0.6 \
  --late_layer_frac 0.0 \
  --score_mode zscore \
  --k_values 1,2,4,8,16,32,64 \
  --random_trials_min 6 \
  --random_trials_max 20 \
  --random_sem_target 0.01 \
  --eval_batch_size 8 \
  --seed 42 \
  --device cuda:1 \
  --steps 1,2,3 \
  --no-save-plots
```

Qwen3-14B / facts-first:

```bash
python3 -m src.heads_analysis.run_all \
  --model_id Qwen/Qwen3-14B \
  --model_source huggingface \
  --output_dir reports/heads_analysis_qwen3_14b_facts_first \
  --input artifacts/filtered_dual_correct_Qwen3-14b.jsonl \
  --hop all \
  --prompt_order facts_first \
  --prompt_style symbolic \
  --token_scope all_tokens \
  --impact_samples 2000 \
  --probe_samples 64 \
  --classify_samples 2000 \
  --validation_samples 1000 \
  --validation_accuracy_samples 500 \
  --top_n 64 \
  --top_m_per_layer 4 \
  --candidate_pool_mult 4 \
  --quantile_keep 0.6 \
  --late_layer_frac 0.0 \
  --score_mode zscore \
  --k_values 1,2,4,8,16,32,64 \
  --random_trials_min 6 \
  --random_trials_max 20 \
  --random_sem_target 0.01 \
  --eval_batch_size 8 \
  --seed 42 \
  --device cuda:1 \
  --steps 1,2,3 \
  --no-save-plots
```

Qwen3-14B / expr-first:

```bash
python3 -m src.heads_analysis.run_all \
  --model_id Qwen/Qwen3-14B \
  --model_source huggingface \
  --output_dir reports/heads_analysis_qwen3_14b_expr_first \
  --input artifacts/filtered_dual_correct_Qwen3-14b_expr_first.jsonl \
  --hop all \
  --prompt_order expr_first \
  --prompt_style symbolic \
  --token_scope all_tokens \
  --impact_samples 2000 \
  --probe_samples 256 \
  --classify_samples 2000 \
  --validation_samples 1000 \
  --validation_accuracy_samples 500 \
  --top_n 64 \
  --top_m_per_layer 4 \
  --candidate_pool_mult 4 \
  --quantile_keep 0.6 \
  --late_layer_frac 0.0 \
  --score_mode zscore \
  --k_values 1,2,4,8,16,32,64 \
  --random_trials_min 6 \
  --random_trials_max 20 \
  --random_sem_target 0.01 \
  --eval_batch_size 8 \
  --seed 42 \
  --device cuda:1 \
  --steps 1,2,3 \
  --no-save-plots
```

### 6.2 再分别画 Step1 / Step2 / Step3 的图

以 `reports/heads_analysis_qwen3_8b_facts_first` 为例：

Step1 图：

```bash
python3 -m src.heads_analysis.plot_step1_heads \
  --classify_csv reports/heads_analysis_qwen3_8b_facts_first/classify/top_heads_pattern_labels.csv \
  --summary_json reports/heads_analysis_qwen3_8b_facts_first/step1_summary.json \
  --output_png reports/heads_analysis_qwen3_8b_facts_first/classify/layer_head_role_distribution.png
```

Step2 图：

```bash
python3 -m src.heads_analysis.plot_step2_taxonomy \
  --counts_csv reports/heads_analysis_qwen3_8b_facts_first/taxonomy/head_taxonomy_counts.csv \
  --output_png reports/heads_analysis_qwen3_8b_facts_first/taxonomy/head_taxonomy_line_chart.png
```

Step3 图：

```bash
python3 -m src.heads_analysis.plot_step3_curves \
  --pd_curve_csv reports/heads_analysis_qwen3_8b_facts_first/validation/pd_curve_metrics.csv \
  --output_dir reports/heads_analysis_qwen3_8b_facts_first/validation/plots
```

其他目录完全同理，只需要把 `reports/heads_analysis_qwen3_8b_facts_first` 替换成对应实验目录即可。

## 7. 清理日志

```bash
bash scripts/clear_logs.sh
```
