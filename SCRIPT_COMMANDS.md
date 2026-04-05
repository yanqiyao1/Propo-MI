# Propo-MI 脚本命令手册

## 1. 项目理解

这个项目是一个面向大语言模型机理研究的命题逻辑基准与分析流水线，主线可以概括为：

1. **构造数据**：生成命题逻辑样本，包含 `one_hop` 与 `two_hop` 两类推理深度。
2. **构造干净/扰动对**：每条样本同时带有 `clean` 与 `corrupted` 两套 facts，用于因果/补丁分析。
3. **模型评测**：让模型分别回答 clean/corrupted prompt，记录预测、正确率和双正确样本。
4. **注意力头机理分析**：筛头、分类头角色、做随机对照验证。
5. **激活补丁分析**：对 residual / attention head / MLP 做逐层或逐头 patching。
6. **区域级 MLP 消融**：按 `facts_region`、`expression_region`、`terminal_token` 三种区域做 MLP 消融。

项目里的关键数据设计如下：

- **推理深度**：`one_hop` 直接判断表达式真假；`two_hop` 先算中间变量，再回答最终表达式。
- **提示布局**：支持 `facts_first` 与 `expr_first`。
- **提示风格**：支持 `symbolic` 与 `semi_natural`。
- **分析对象**：既看 clean/corrupted 对的行为差异，也看补丁后 True/False 概率差（dPD）的变化。

## 2. 运行约定

- 所有命令都建议在仓库根目录执行：`/home/qiyaoyan/Propo-MI`
- 统一使用模块方式运行：`python -m ...`
- 主要依赖从源码可见包括：`torch`、`modelscope`、`transformers`、`huggingface_hub`、`transformer_lens`、`numpy`、`matplotlib`
- `heads_analysis` 与 `src/mech`、`src/mlp_analysis` 都依赖 `transformer_lens`
- 所有带 `--model_id` 的模型加载脚本现在都支持 `--model_source modelscope|huggingface`
- `--model_source` 默认是 `modelscope`；如果想直接从 HuggingFace 加载，显式传 `--model_source huggingface`
- `heads_analysis` 的 `--input` 留空时，会自动回退到：
  - `artifacts/filtered_dual_correct_14b.jsonl`：当 `--model_id` 包含 `14B`
  - `artifacts/filtered_dual_correct_8b.jsonl`：其他情况
- 清理仓库内所有日志文件可直接运行：`bash scripts/clear_logs.sh`

### 2.1 进度与日志

- 现在实验主流程默认用 `tqdm` 在终端显示进度条；原来的阶段性 `print` 与摘要输出会写入日志文件。
- **单输出文件**脚本通常把日志写到输出文件旁边：
  - `artifacts/preds_8b_nocot.jsonl` → `artifacts/preds_8b_nocot.log`
  - `reports/metrics_8b_nocot.json` → `reports/metrics_8b_nocot.log`
- **输出目录**脚本通常把日志写到输出目录内固定文件名，便于后续排查。

**常见日志文件速查**

| 模块 | 日志位置 |
| --- | --- |
| `src.data.generate_dataset` | `<output>.log` |
| `src.data.split_by_hop` | 若传 `--summary`：`<summary同名>.log`；否则：`<out_one>.log` |
| `src.eval.inference` | `<output>.log` |
| `src.eval.filtering` | `<output>.log` |
| `src.eval.metrics` | 若传 `--output`：`<output>.log`；否则：`<input>.metrics.log` |
| `src.heads_analysis.step1_discover_fast` | `<output_dir>/step1_discover_fast.log` |
| `src.heads_analysis.step2_taxonomy` | `<output_dir>/step2_taxonomy.log` |
| `src.heads_analysis.step3_validate_fast` | `<output_dir>/step3_validate_fast.log` |
| `src.heads_analysis.run_all` | `<output_dir>/run_all.log` |
| `src.mech.patch_residual` | `<output>.log` |
| `src.mech.patch_heads` | `<output>.log` |
| `src.mech.patch_mlp` | `<output>.log` |
| `src.mlp_analysis.mlp_region_ablation` | `<output_dir>/mlp_region_ablation.log` |
| `src.mlp_analysis.plot_band_metrics` | `<output_dir>/plot_band_metrics.log` |
| `src.token_analysis.activation_patching_dataset` | `<output_dir>/activation_patching_dataset.log` |
| `src.token_analysis.refined_token_analysis` | `<output_dir>/refined_token_analysis.log` |

## 3. CLI 脚本总表

| 模块 | 功能 |
| --- | --- |
| `src.data.generate_dataset` | 生成 PropLogic-MI 数据集 |
| `src.data.split_by_hop` | 将数据拆成 one-hop / two-hop |
| `src.eval.inference` | 对 clean / corrupted prompt 做推理 |
| `src.eval.filtering` | 过滤 dual-correct 样本 |
| `src.eval.metrics` | 统计准确率与前后对比指标 |
| `src.heads_analysis.step1_discover_fast` | 注意力头筛选 + 角色分类 |
| `src.heads_analysis.step2_taxonomy` | 基于 Step1 结果绘制 taxonomy 图 |
| `src.heads_analysis.step3_validate_fast` | 头角色 necessity / sufficiency 验证 |
| `src.heads_analysis.run_all` | Step1~3 总入口 |
| `src.mech.patch_residual` | residual stream patching 扫描 |
| `src.mech.patch_heads` | attention head patching 扫描 |
| `src.mech.patch_mlp` | MLP patching 扫描 |
| `src.mlp_analysis.mlp_region_ablation` | 面向布局区域的 MLP 消融 |

> 说明：`src.data.logic_ast`、`src.data.formatters`、`src.data.template_rules`、`src.mech.dld`、`src.mech.head_taxonomy`、`src.heads_analysis.common`、`src.mlp_analysis.common` 等属于支撑模块，不是直接命令行入口。

## 4. 数据构造脚本

### 4.1 生成数据集：`src.data.generate_dataset`

**完整命令模板**

```bash
python -m src.data.generate_dataset \
  --output <输出jsonl> \
  [--per_rule_per_hop 200] \
  [--target_count 0] \
  [--one_hop_ratio 0.5] \
  [--max_attempts_multiplier 200] \
  [--seed 42] \
  [--prompt_order facts_first|expr_first] \
  [--prompt_ending answer_suffix|terminal_is]
```

**用途**

- 生成带 `clean/corrupted` prompt 的命题逻辑数据
- 输出样本中包含 `facts`、`corrupted_facts`、`label`、`label_corrupted`、prompt 文本等字段，不再写入 `split` 字段
- `--prompt_ending terminal_is` 会让 prompt 直接以最终的 `is` 结尾，不再追加 `Answer with one word...`；当前需配合 `--prompt_order facts_first`
- 规则模板来自命题逻辑恒等律、德摩根律、分配律、结合律、吸收律等

**常用示例**

```bash
python -m src.data.generate_dataset \
  --output dataset/proplogic_mi.jsonl \
  --target_count 10000 \
  --one_hop_ratio 0.5 \
  --seed 42 \
  --prompt_order facts_first
```

```bash
python -m src.data.generate_dataset \
  --output dataset/proplogic_mi_terminal_is.jsonl \
  --target_count 2000 \
  --seed 42 \
  --prompt_order facts_first \
  --prompt_ending terminal_is
```

```bash
python -m src.data.generate_dataset \
  --output dataset/proplogic_mi_expr_first_new.jsonl \
  --target_count 20000 \
  --seed 42 \
  --prompt_order expr_first
```

### 4.2 按 hop 拆分：`src.data.split_by_hop`

**完整命令模板**

```bash
python -m src.data.split_by_hop \
  --input <输入jsonl> \
  --out_one <one_hop输出jsonl> \
  --out_two <two_hop输出jsonl> \
  [--summary <统计json>]
```

**常用示例**

```bash
python -m src.data.split_by_hop \
  --input artifacts/filtered_dual_correct_14b.jsonl \
  --out_one dataset/proplogic_mi_one_hop.jsonl \
  --out_two dataset/proplogic_mi_two_hop.jsonl \
  --summary reports/mlp_analysis/dataset_hop_summary.json
```

## 5. 模型评测脚本

### 5.1 推理：`src.eval.inference`

**完整命令模板**

```bash
python -m src.eval.inference \
  --model_id <模型名或本地路径> \
  [--model_source modelscope|huggingface] \
  --input <输入jsonl> \
  --output <输出jsonl> \
  [--prompt_style symbolic|semi_natural] \
  [--mode nocot|cot] \
  [--max_new_tokens 8] \
  [--temperature 0.0] \
  [--top_p 0.95] \
  [--dtype auto|float16|bfloat16|float32] \
  [--device_map auto|balanced|sequential|none] \
  [--device auto|cuda:0|cpu] \
  [--max_samples 0] \
  [--enable_thinking auto|true|false] \
  [--progress_every 50]
```

**用途**

- 分别对 `clean_prompt_*` 与 `corrupted_prompt_*` 运行生成
- 自动解析输出中的 `True/False`
- 在输出样本中增加 `pred_clean`、`pred_corrupted`、`correct_clean`、`correct_corrupted`
- 默认从 `ModelScope` 加载；可通过 `--model_source huggingface` 切换到 HuggingFace

**常用示例**

```bash
python -m src.eval.inference \
  --model_id Qwen/Qwen3-8B \
  --model_source huggingface \
  --input dataset/proplogic_mi.jsonl \
  --output artifacts/preds_8b_nocot_test.jsonl \
  --prompt_style symbolic \
  --max_new_tokens 1 \
  --mode nocot \
  --temperature 0.0 \
  --max_samples 100 \
  --dtype auto \
  --device cuda:0
```

```bash
python -m src.eval.inference \
  --model_id Qwen/Qwen3-14B \
  --input dataset/proplogic_mi_expr_first.jsonl \
  --output artifacts/preds_14b_expr_first.jsonl \
  --prompt_style symbolic \
  --mode nocot
```

### 5.2 双正确过滤：`src.eval.filtering`

**完整命令模板**

```bash
python -m src.eval.filtering \
  --input <推理输出jsonl> \
  --output <过滤后的jsonl> \
  [--require_label_change]
```

**用途**

- 只保留 `correct_clean=True` 且 `correct_corrupted=True` 的样本
- 可选要求 `label` 与 `label_corrupted` 必须变化，更适合某些因果补丁分析

**常用示例**

```bash
python -m src.eval.filtering \
  --input artifacts/preds_8b_nocot.jsonl \
  --output artifacts/filtered_dual_correct_8b.jsonl
```

```bash
python -m src.eval.filtering \
  --input artifacts/preds_14b_expr_first_test.jsonl \
  --output artifacts/filtered_14b_expr_first_test_changed.jsonl \
  --require_label_change
```

### 5.3 指标统计：`src.eval.metrics`

**完整命令模板**

```bash
python -m src.eval.metrics \
  --input <输入jsonl> \
  [--output <输出json>] \
  [--after <对比jsonl>]
```

**用途**

- 统计 clean / corrupted 正确率、dual-correct rate
- 可按 `hop`、`rule` 聚合
- 可比较 before / after 两份预测结果，输出 error flip rate 与 regression rate

**常用示例**

```bash
python -m src.eval.metrics \
  --input artifacts/preds_8b_nocot.jsonl \
  --output reports/metrics_8b_nocot.json
```

```bash
python -m src.eval.metrics \
  --input artifacts/preds_14b_expr_first_test.jsonl \
  --output reports/metrics_14b_expr_first_test.json
```

```bash
python -m src.eval.metrics \
  --input before.jsonl \
  --after after.jsonl \
  --output reports/before_after.json
```

## 6. 注意力头机理分析脚本

### 6.1 Step1：头筛选与角色分类 `src.heads_analysis.step1_discover_fast`

**完整命令模板**

```bash
python -m src.heads_analysis.step1_discover_fast \
  --model_id <模型名或本地路径> \
  --output_dir <输出目录> \
  [--input <dual-correct jsonl>] \
  [--hop one_hop|two_hop|all] \
  [--prompt_order facts_first|expr_first|all] \
  [--prompt_style symbolic|semi_natural] \
  [--impact_samples 500] \
  [--probe_samples 64] \
  [--classify_samples 500] \
  [--top_n 64] \
  [--top_m_per_layer 4] \
  [--candidate_pool_mult 4] \
  [--quantile_keep 0.6] \
  [--late_layer_frac 0.0] \
  [--token_scope query_only|all_tokens] \
  [--score_mode zscore] \
  [--eval_batch_size 16] \
  [--seed 42] \
  [--device cuda] \
  [--progress_every 10]
```

**输出重点**

- `impact/impact_probe_all_heads.csv`
- `impact/impact_refined_candidates.csv`
- `impact/impact_top_heads.csv`
- `impact/impact_top_heads.json`
- `classify/top_heads_pattern_labels.csv`
- `classify/top_heads_pattern_labels.json`
- `classify/layer_head_role_distribution.png`
- `step1_summary.json`

**常用示例**

```bash
python -m src.heads_analysis.step1_discover_fast \
  --model_id Qwen/Qwen3-8B \
  --input artifacts/filtered_dual_correct_8b.jsonl \
  --output_dir reports/heads_8b \
  --hop all \
  --prompt_order facts_first \
  --prompt_style symbolic
```

### 6.2 Step2：taxonomy 绘图 `src.heads_analysis.step2_taxonomy`

**完整命令模板**

```bash
python -m src.heads_analysis.step2_taxonomy \
  --classify_csv <Step1输出csv> \
  --output_dir <输出目录> \
  [--n_layers 0] \
  [--role_col role_label]
```

**常用示例**

```bash
python -m src.heads_analysis.step2_taxonomy \
  --classify_csv reports/heads_8b/classify/top_heads_pattern_labels.csv \
  --output_dir reports/heads_8b/taxonomy
```

### 6.3 Step3：角色验证 `src.heads_analysis.step3_validate_fast`

**完整命令模板**

```bash
python -m src.heads_analysis.step3_validate_fast \
  --model_id <模型名或本地路径> \
  --classify_json <Step1输出json> \
  --output_dir <输出目录> \
  [--input <dual-correct jsonl>] \
  [--hop one_hop|two_hop|all] \
  [--prompt_order facts_first|expr_first|all] \
  [--prompt_style symbolic|semi_natural] \
  [--max_samples 400] \
  [--k_values 1,2,4,8,16] \
  [--late_layer_frac 0.0] \
  [--token_scope query_only|all_tokens] \
  [--random_trials_min 6] \
  [--random_trials_max 20] \
  [--random_sem_target 0.01] \
  [--eval_batch_size 8] \
  [--seed 42] \
  [--device cuda] \
  [--progress_every 10]
```

**输出重点**

- `condition_level.csv`
- `plots/necessity_topk_vs_random_curve.png`
- `plots/sufficiency_recovery_curve.png`
- `summary.json`

**常用示例**

```bash
python -m src.heads_analysis.step3_validate_fast \
  --model_id Qwen/Qwen3-8B \
  --classify_json reports/heads_8b/classify/top_heads_pattern_labels.json \
  --output_dir reports/heads_8b/validation \
  --input artifacts/filtered_dual_correct_8b.jsonl \
  --hop all \
  --prompt_order facts_first \
  --prompt_style symbolic
```

### 6.4 一键跑完整头分析：`src.heads_analysis.run_all`

**完整命令模板**

```bash
python -m src.heads_analysis.run_all \
  --model_id <模型名或本地路径> \
  --output_dir <输出目录> \
  [--input <dual-correct jsonl>] \
  [--hop one_hop|two_hop|all] \
  [--prompt_order facts_first|expr_first|all] \
  [--prompt_style symbolic|semi_natural] \
  [--impact_samples 500] \
  [--probe_samples 64] \
  [--classify_samples 500] \
  [--top_n 64] \
  [--top_m_per_layer 4] \
  [--candidate_pool_mult 4] \
  [--quantile_keep 0.6] \
  [--late_layer_frac 0.0] \
  [--token_scope query_only|all_tokens] \
  [--score_mode zscore] \
  [--eval_batch_size 8] \
  [--validation_samples 400] \
  [--k_values 1,2,4,8,16] \
  [--random_trials_min 6] \
  [--random_trials_max 20] \
  [--random_sem_target 0.01] \
  [--seed 42] \
  [--device cuda] \
  [--progress_every 10] \
  [--steps 1,2,3]
```

**常用示例**

```bash
python -m src.heads_analysis.run_all \
  --model_id Qwen/Qwen3-14B \
  --input artifacts/filtered_dual_correct_14b.jsonl \
  --output_dir reports/heads_14b \
  --hop all \
  --prompt_order facts_first \
  --prompt_style symbolic
```

```bash
python -m src.heads_analysis.run_all \
  --model_id Qwen/Qwen3-8B \
  --output_dir reports/heads_8b_only_step1 \
  --steps 1
```

## 7. 直接激活补丁脚本

这些脚本通常建议输入 `dual-correct` 数据，以保证 clean/corrupted 两边都是模型原本答对的，再看补丁造成的 dPD 变化。

### 7.1 Residual patching：`src.mech.patch_residual`

```bash
python -m src.mech.patch_residual \
  --model_id <模型名或本地路径> \
  --input <输入jsonl> \
  --output <输出csv> \
  [--prompt_style symbolic|semi_natural] \
  [--max_samples 64] \
  [--device cuda] \
  [--patch_position query]
```

**示例**

```bash
python -m src.mech.patch_residual \
  --model_id Qwen/Qwen3-8B \
  --input artifacts/filtered_dual_correct_8b.jsonl \
  --output reports/mech/residual_patch_8b.csv
```

### 7.2 Head patching：`src.mech.patch_heads`

```bash
python -m src.mech.patch_heads \
  --model_id <模型名或本地路径> \
  --input <输入jsonl> \
  --output <输出csv> \
  [--prompt_style symbolic|semi_natural] \
  [--max_samples 32] \
  [--device cuda]
```

**示例**

```bash
python -m src.mech.patch_heads \
  --model_id Qwen/Qwen3-8B \
  --input artifacts/filtered_dual_correct_8b.jsonl \
  --output reports/mech/head_patch_8b.csv
```

### 7.3 MLP patching：`src.mech.patch_mlp`

```bash
python -m src.mech.patch_mlp \
  --model_id <模型名或本地路径> \
  --input <输入jsonl> \
  --output <输出csv> \
  [--prompt_style symbolic|semi_natural] \
  [--max_samples 64] \
  [--device cuda]
```

**示例**

```bash
python -m src.mech.patch_mlp \
  --model_id Qwen/Qwen3-14B \
  --input artifacts/filtered_dual_correct_14b.jsonl \
  --output reports/mech/mlp_patch_14b.csv
```

## 8. 区域级 MLP 消融脚本

### 8.1 区域 MLP 消融：`src.mlp_analysis.mlp_region_ablation`

**完整命令模板**

```bash
python -m src.mlp_analysis.mlp_region_ablation \
  --model_id <模型名或本地路径> \
  --input <输入jsonl> \
  --output_dir <输出目录> \
  [--prompt_style symbolic|semi_natural] \
  --region_mode facts_region|expression_region|terminal_token \
  [--max_samples 0] \
  [--device cuda] \
  [--eps 1e-6] \
  [--progress_every 100]
```

**用途**

- 不是把 clean 激活 patch 到 corrupt，而是对指定 token 区域的 MLP 输出做 mean-patching
- 输出每层的 patching score、summary 和柱状图

**常用示例**

```bash
python -m src.mlp_analysis.mlp_region_ablation \
  --model_id Qwen/Qwen3-14B \
  --input dataset/proplogic_mi_one_hop.jsonl \
  --output_dir reports/mlp_analysis/14b_onehop_facts \
  --prompt_style symbolic \
  --region_mode facts_region \
  --max_samples 1
  --device cuda:6
```

```bash
python -m src.mlp_analysis.mlp_region_ablation \
  --model_id Qwen/Qwen3-14B \
  --input dataset/proplogic_mi_two_hop.jsonl \
  --output_dir reports/mlp_analysis/14b_twohop_expr \
  --prompt_style symbolic \
  --region_mode expression_region \
  --max_samples 500
```

```bash
python -m src.mlp_analysis.mlp_region_ablation \
  --model_id Qwen/Qwen3-8B \
  --input artifacts/filtered_dual_correct_8b.jsonl \
  --output_dir reports/mlp_analysis/8b_terminal \
  --region_mode terminal_token \
  --max_samples 300
```

## 9. 推荐复现实验顺序

### 路线 A：从头构造并评测

```bash
python -m src.data.generate_dataset --output dataset/proplogic_mi_new.jsonl --target_count 20000 --prompt_order facts_first
# 如果想让 prompt 直接以 is 结尾：
# python -m src.data.generate_dataset --output dataset/proplogic_mi_terminal_is.jsonl --target_count 20000 --prompt_order facts_first --prompt_ending terminal_is
python -m src.eval.inference --model_id Qwen/Qwen3-8B --input dataset/proplogic_mi_new.jsonl --output artifacts/preds_8b_new.jsonl --prompt_style symbolic --mode nocot
python -m src.eval.metrics --input artifacts/preds_8b_new.jsonl --output reports/metrics_8b_new.json
python -m src.eval.filtering --input artifacts/preds_8b_new.jsonl --output artifacts/filtered_dual_correct_8b_new.jsonl
python -m src.heads_analysis.run_all --model_id Qwen/Qwen3-8B --input artifacts/filtered_dual_correct_8b_new.jsonl --output_dir reports/heads_8b_new
```

### 路线 B：直接基于仓库现成产物做机理分析

```bash
python -m src.heads_analysis.run_all --model_id Qwen/Qwen3-14B --input artifacts/filtered_dual_correct_14b.jsonl --output_dir reports/heads_14b
python -m src.mech.patch_heads --model_id Qwen/Qwen3-14B --input artifacts/filtered_dual_correct_14b.jsonl --output reports/mech/head_patch_14b.csv
python -m src.mech.patch_mlp --model_id Qwen/Qwen3-14B --input artifacts/filtered_dual_correct_14b.jsonl --output reports/mech/mlp_patch_14b.csv
python -m src.data.split_by_hop --input artifacts/filtered_dual_correct_14b.jsonl --out_one dataset/proplogic_mi_one_hop.jsonl --out_two dataset/proplogic_mi_two_hop.jsonl --summary reports/mlp_analysis/dataset_hop_summary.json
python -m src.mlp_analysis.mlp_region_ablation --model_id Qwen/Qwen3-14B --input dataset/proplogic_mi_one_hop.jsonl --output_dir reports/mlp_analysis/14b_onehop_facts --region_mode facts_region --max_samples 500
```

## 10. 快速查看帮助

每个 CLI 脚本都支持：

```bash
python -m <模块路径> --help
```

例如：

```bash
python -m src.eval.inference --help
python -m src.heads_analysis.run_all --help
python -m src.mlp_analysis.mlp_region_ablation --help
```
