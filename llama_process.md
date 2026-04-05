本文档记录 `meta-llama/Llama-3.1-8B-Instruct` 的完整实验流程。

所有输出都与原有 Qwen3 实验隔离：

- 预测结果与过滤后的样本写入 `artifacts_llama3/`
- 分析报告写入 `reports_llama3/`
- MLP 按 hop 切分后的中间数据写入 `dataset_llama3/`

推荐模型名：

```bash
MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
```

推荐使用本地目录方案。你已经可以先用 ModelScope 下载，然后把本地目录传给本项目。

模型下载命令：

```bash
modelscope download \
  --model LLM-Research/Meta-Llama-3.1-8B-Instruct \
  --local_dir /media/snail-ssd/models/Llama-3.1-8B-Instruct
```

下载完成后，本项目中统一使用这个本地目录：

```bash
MODEL_ID="/media/snail-ssd/models/Llama-3.1-8B-Instruct"
```

这样做的好处是：

- 避开 HuggingFace gated repo 的在线拉取限制
- 推理和 `TransformerLens` 分析都直接复用同一个本地目录
- 实验复现更稳定

推荐的设备分配方式：

- `INFERENCE_DEVICE`：`transformers` 推理阶段使用的 GPU
- `HEADS_DEVICE`：`TransformerLens` 的 heads analysis 使用的 GPU
- `MLP_DEVICE`：`TransformerLens` 的 MLP analysis 使用的 GPU
- `TOKEN_DEVICE`：`TransformerLens` 的 token analysis 使用的 GPU

如果你只有一张卡，就把下面所有设备都改成同一张卡。

示例：

```bash
INFERENCE_DEVICE="cuda:0"
HEADS_DEVICE="cuda:1"
MLP_DEVICE="cuda:1"
TOKEN_DEVICE="cuda:1"
```

你现在可以直接用一条命令跑完整 Llama 流程：

```bash
MODEL_ID=/media/snail-ssd/models/Llama-3.1-8B-Instruct \
INFERENCE_DEVICE=cuda:0 \
HEADS_DEVICE=cuda:1 \
MLP_DEVICE=cuda:1 \
TOKEN_DEVICE=cuda:1 \
bash scripts/run_llama3_full.sh
```

如果你更想分阶段手动执行，可以使用下面的详细命令。

## 0. 生成评测数据集

Facts-first 数据集：

```bash
python -m src.data.generate_dataset \
  --output dataset/proplogic_mi.jsonl \
  --target_count 10000 \
  --one_hop_ratio 0.5 \
  --seed 42 \
  --prompt_order facts_first
```

Expr-first 数据集：

```bash
python -m src.data.generate_dataset \
  --output dataset/proplogic_mi_expr_first.jsonl \
  --target_count 10000 \
  --one_hop_ratio 0.5 \
  --seed 42 \
  --prompt_order expr_first
```

## 1. 推理

Facts-first：

```bash
python -m src.eval.inference \
  --model_id /media/snail-ssd/models/Llama-3.1-8B-Instruct \
  --model_source huggingface \
  --input dataset/proplogic_mi.jsonl \
  --output artifacts_llama3/preds_facts_first.jsonl \
  --prompt_style symbolic \
  --max_new_tokens 1 \
  --mode nocot \
  --temperature 0.0 \
  --max_samples 0 \
  --dtype auto \
  --device cuda:0
```

Expr-first：

```bash
python -m src.eval.inference \
  --model_id /media/snail-ssd/models/Llama-3.1-8B-Instruct \
  --model_source huggingface \
  --input dataset/proplogic_mi_expr_first.jsonl \
  --output artifacts_llama3/preds_expr_first.jsonl \
  --prompt_style symbolic \
  --max_new_tokens 1 \
  --mode nocot \
  --temperature 0.0 \
  --max_samples 0 \
  --dtype auto \
  --device cuda:0
```

## 2. 过滤 dual-correct 样本

Facts-first：

```bash
python -m src.eval.filtering \
  --input artifacts_llama3/preds_facts_first.jsonl \
  --output artifacts_llama3/filtered_dual_correct_facts_first.jsonl
```

Expr-first：

```bash
python -m src.eval.filtering \
  --input artifacts_llama3/preds_expr_first.jsonl \
  --output artifacts_llama3/filtered_dual_correct_expr_first.jsonl
```

## 3. 计算指标

Facts-first：

```bash
python -m src.eval.metrics \
  --input artifacts_llama3/preds_facts_first.jsonl \
  --output reports_llama3/metrics_facts_first.json
```

Expr-first：

```bash
python -m src.eval.metrics \
  --input artifacts_llama3/preds_expr_first.jsonl \
  --output reports_llama3/metrics_expr_first.json
```

## 4. MLP region patching

Facts-first：

```bash
bash scripts/MLP_region_patching.sh \
  --model_id "/media/snail-ssd/models/Llama-3.1-8B-Instruct" \
  --model_source "huggingface" \
  --prompt_style "symbolic" \
  --device "cuda:1" \
  --max_samples 1000 \
  --progress_every 100 \
  --split "all" \
  --source_input "artifacts_llama3/filtered_dual_correct_facts_first.jsonl" \
  --one_hop_input "dataset_llama3/mlp_one_hop_facts_first.jsonl" \
  --two_hop_input "dataset_llama3/mlp_two_hop_facts_first.jsonl" \
  --split_summary "reports_llama3/mlp_analysis_facts_first/dual_correct_split.json" \
  --output_root "reports_llama3/mlp_analysis_facts_first" \
  --auto_split 1 \
  --make_plots 1 \
  --plot_output_dir "reports_llama3/mlp_analysis_facts_first/comparison"
```

Expr-first：

```bash
bash scripts/MLP_region_patching.sh \
  --model_id "/media/snail-ssd/models/Llama-3.1-8B-Instruct" \
  --model_source "huggingface" \
  --prompt_style "symbolic" \
  --device "cuda:1" \
  --max_samples 1000 \
  --progress_every 100 \
  --split "all" \
  --source_input "artifacts_llama3/filtered_dual_correct_expr_first.jsonl" \
  --one_hop_input "dataset_llama3/mlp_one_hop_expr_first.jsonl" \
  --two_hop_input "dataset_llama3/mlp_two_hop_expr_first.jsonl" \
  --split_summary "reports_llama3/mlp_analysis_expr_first/dual_correct_split.json" \
  --output_root "reports_llama3/mlp_analysis_expr_first" \
  --auto_split 1 \
  --make_plots 1 \
  --plot_output_dir "reports_llama3/mlp_analysis_expr_first/comparison"
```

## 5. Token analysis

当前对 Llama-3.1-8B 使用的阶段划分为：

- `early_end=12`
- `middle_end=22`

Facts-first 原始 patching：

```bash
python3 -m src.token_analysis.activation_patching_dataset \
  --model_id /media/snail-ssd/models/Llama-3.1-8B-Instruct \
  --model_source huggingface \
  --input artifacts_llama3/filtered_dual_correct_facts_first.jsonl \
  --output_dir reports_llama3/token_analysis_facts_first_raw \
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

Facts-first 精细分析：

```bash
python3 -m src.token_analysis.refined_token_analysis \
  --input_pkl reports_llama3/token_analysis_facts_first_raw/patching_results.pkl \
  --output_dir reports_llama3/token_analysis_facts_first_refined \
  --title "Llama-3.1-8B-Instruct Facts-first All-hop Refined Token Analysis" \
  --early_end 12 \
  --middle_end 22 \
  --n_layers 0 \
  --include-derived-assignment \
  --save-plots \
  --save-csv
```

Expr-first 原始 patching：

```bash
python3 -m src.token_analysis.activation_patching_dataset \
  --model_id /media/snail-ssd/models/Llama-3.1-8B-Instruct \
  --model_source huggingface \
  --input artifacts_llama3/filtered_dual_correct_expr_first.jsonl \
  --output_dir reports_llama3/token_analysis_expr_first_raw \
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

Expr-first 精细分析：

```bash
python3 -m src.token_analysis.refined_token_analysis \
  --input_pkl reports_llama3/token_analysis_expr_first_raw/patching_results.pkl \
  --output_dir reports_llama3/token_analysis_expr_first_refined \
  --title "Llama-3.1-8B-Instruct Expr-first All-hop Refined Token Analysis" \
  --early_end 12 \
  --middle_end 22 \
  --n_layers 0 \
  --include-derived-assignment \
  --save-plots \
  --save-csv
```

## 6. Heads analysis

Facts-first：

```bash
python3 -m src.heads_analysis.run_all \
  --model_id /media/snail-ssd/models/Llama-3.1-8B-Instruct \
  --model_source huggingface \
  --output_dir reports_llama3/heads_analysis_facts_first \
  --input artifacts_llama3/filtered_dual_correct_facts_first.jsonl \
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

Expr-first：

```bash
python3 -m src.heads_analysis.run_all \
  --model_id /media/snail-ssd/models/Llama-3.1-8B-Instruct \
  --model_source huggingface \
  --output_dir reports_llama3/heads_analysis_expr_first \
  --input artifacts_llama3/filtered_dual_correct_expr_first.jsonl \
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

## 7. 使用封装脚本按阶段重跑

如果你想保持输出目录隔离，但使用封装好的 runner，下面是最常用的几种方式。

运行完整 Llama 流程：

```bash
MODEL_ID=/media/snail-ssd/models/Llama-3.1-8B-Instruct \
INFERENCE_DEVICE=cuda:0 \
HEADS_DEVICE=cuda:1 \
MLP_DEVICE=cuda:1 \
TOKEN_DEVICE=cuda:1 \
bash scripts/run_llama3_full.sh
```

只跑 inference、filtering 和 metrics：

```bash
RUN_HEADS=0 \
RUN_MLP=0 \
RUN_TOKEN=0 \
MODEL_ID=/media/snail-ssd/models/Llama-3.1-8B-Instruct \
INFERENCE_DEVICE=cuda:0 \
bash scripts/run_llama3_full.sh
```

前面的推理都已经完成，只重跑重型 `TransformerLens` 阶段：

```bash
RUN_INFERENCE=0 \
RUN_FILTERING=0 \
RUN_METRICS=0 \
MODEL_ID=/media/snail-ssd/models/Llama-3.1-8B-Instruct \
HEADS_DEVICE=cuda:1 \
MLP_DEVICE=cuda:1 \
TOKEN_DEVICE=cuda:1 \
bash scripts/run_llama3_full.sh
```

## 8. 清理日志

```bash
bash scripts/clear_logs.sh
```
