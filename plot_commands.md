# 所有实验的画图命令汇总

本文专门汇总当前项目里所有“单独画图”的命令。

适用范围：

- Qwen3-8B
- Qwen3-14B
- Mistral-7B-Instruct-v0.1
- Llama-3.1-8B-Instruct
- facts-first / expr-first

注意：这些命令都默认建立在“结果文件已经先跑出来”的前提下。
也就是你已经先完成了对应实验的结果生成阶段，并且目录里已经有 `csv/json/pkl` 文件。

---

## 1. MLP analysis 画图命令

MLP 实验有三类图：

1. 每个 region 的逐层柱状图
2. 每个 region 的 `|dPD|` 逐层柱状图
3. one-hop / two-hop 的 band comparison 图

### 1.1 单个 region 的逐层柱状图

命令模板：

```bash
python -m src.mlp_analysis.plot_region_scores \
  --scores_csv <某个region目录下的csv> \
  --output_png <想保存的png路径>
```

如果要画 `|dPD|` 作为 patching score 的图，增加：

```bash
  --score_column abs_dpd
```

四个 region 分别是：

- `facts_region`
- `expression_region`
- `constrain_region`
- `terminal_token`

示例：Qwen3-8B, facts-first, one-hop, facts_region

```bash
python -m src.mlp_analysis.plot_region_scores \
  --scores_csv reports/mlp_analysis/mlp_8b_one_hop/facts_region/mlp_facts_region_scores.csv \
  --output_png reports/mlp_analysis/mlp_8b_one_hop/facts_region/mlp_facts_region_scores.png
```

示例：Mistral, expr-first, two-hop, expression_region

```bash
python -m src.mlp_analysis.plot_region_scores \
  --scores_csv reports_mistral/mlp_analysis_expr_first/mlp_two_hop_expr_first/expression_region/mlp_expression_region_scores.csv \
  --output_png reports_mistral/mlp_analysis_expr_first/mlp_two_hop_expr_first/expression_region/mlp_expression_region_scores.png
```

示例：Qwen3-8B, facts-first, one-hop, facts_region, 画 `|dPD|` 图

```bash
python -m src.mlp_analysis.plot_region_scores \
  --scores_csv reports/mlp_analysis/mlp_8b_one_hop/facts_region/mlp_facts_region_scores.csv \
  --score_column abs_dpd \
  --output_png reports/mlp_analysis/mlp_8b_one_hop/facts_region/mlp_facts_region_abs_dpd_scores.png
```

### 1.2 MLP band comparison 图

命令模板：

```bash
python -m src.mlp_analysis.plot_band_metrics \
  --one_hop_dir <one-hop聚合目录> \
  --two_hop_dir <two-hop聚合目录> \
  --output_dir <comparison输出目录>
```

会生成：

- `band_metrics_panel.png`
- `band_metrics_panel.pdf`
- `bcr_stacked.png`
- `bcr_stacked.pdf`

#### Qwen3-8B

facts-first:

```bash
python -m src.mlp_analysis.plot_band_metrics \
  --one_hop_dir reports/mlp_analysis/mlp_8b_one_hop \
  --two_hop_dir reports/mlp_analysis/mlp_8b_two_hop \
  --output_dir reports/mlp_analysis/comparison
```

expr-first:

```bash
python -m src.mlp_analysis.plot_band_metrics \
  --one_hop_dir reports/mlp_analysis_expr_first/mlp_8b_one_hop_expr_first \
  --two_hop_dir reports/mlp_analysis_expr_first/mlp_8b_two_hop_expr_first \
  --output_dir reports/mlp_analysis_expr_first/comparison_expr_first
```

#### Qwen3-14B

facts-first:

```bash
python -m src.mlp_analysis.plot_band_metrics \
  --one_hop_dir reports/mlp_analysis/mlp_14b_one_hop \
  --two_hop_dir reports/mlp_analysis/mlp_14b_two_hop \
  --output_dir reports/mlp_analysis/comparison_14B
```

expr-first:

```bash
python -m src.mlp_analysis.plot_band_metrics \
  --one_hop_dir reports/mlp_analysis_expr_first/mlp_14b_one_hop_expr_first \
  --two_hop_dir reports/mlp_analysis_expr_first/mlp_14b_two_hop_expr_first \
  --output_dir reports/mlp_analysis_expr_first/comparison_expr_first_14B
```

#### Mistral-7B-Instruct-v0.1

facts-first:

```bash
python -m src.mlp_analysis.plot_band_metrics \
  --one_hop_dir reports_mistral/mlp_analysis_facts_first/mlp_one_hop_facts_first \
  --two_hop_dir reports_mistral/mlp_analysis_facts_first/mlp_two_hop_facts_first \
  --output_dir reports_mistral/mlp_analysis_facts_first/comparison
```

expr-first:

```bash
python -m src.mlp_analysis.plot_band_metrics \
  --one_hop_dir reports_mistral/mlp_analysis_expr_first/mlp_one_hop_expr_first \
  --two_hop_dir reports_mistral/mlp_analysis_expr_first/mlp_two_hop_expr_first \
  --output_dir reports_mistral/mlp_analysis_expr_first/comparison
```

#### Llama-3.1-8B-Instruct

facts-first:

```bash
python -m src.mlp_analysis.plot_band_metrics \
  --one_hop_dir reports_llama3/mlp_analysis_facts_first/mlp_one_hop_facts_first \
  --two_hop_dir reports_llama3/mlp_analysis_facts_first/mlp_two_hop_facts_first \
  --output_dir reports_llama3/mlp_analysis_facts_first/comparison
```

expr-first:

```bash
python -m src.mlp_analysis.plot_band_metrics \
  --one_hop_dir reports_llama3/mlp_analysis_expr_first/mlp_one_hop_expr_first \
  --two_hop_dir reports_llama3/mlp_analysis_expr_first/mlp_two_hop_expr_first \
  --output_dir reports_llama3/mlp_analysis_expr_first/comparison
```

---

## 2. Token analysis 画图命令

Token analysis 分两类图：

1. simple 图：直接根据 raw patching 结果画
2. refined 图：根据 refined stats json 画

### 2.1 simple 图

命令模板：

```bash
python3 -m src.token_analysis.plot_simple_analysis \
  --input_pkl <raw目录里的patching_results.pkl> \
  --summary_json <raw目录里的summary.json> \
  --output_dir <raw目录>
```

会生成：

- `category_comparison_simple.png`
- `layer_stage_simple.png`
- `heatmap_simple.png`

#### Qwen3-8B

facts-first:

```bash
python3 -m src.token_analysis.plot_simple_analysis \
  --input_pkl reports/token_analysis/8b_facts_first_all_hop_raw/patching_results.pkl \
  --summary_json reports/token_analysis/8b_facts_first_all_hop_raw/summary.json \
  --output_dir reports/token_analysis/8b_facts_first_all_hop_raw
```

expr-first:

```bash
python3 -m src.token_analysis.plot_simple_analysis \
  --input_pkl reports/token_analysis/8b_expr_first_all_hop_raw/patching_results.pkl \
  --summary_json reports/token_analysis/8b_expr_first_all_hop_raw/summary.json \
  --output_dir reports/token_analysis/8b_expr_first_all_hop_raw
```

#### Qwen3-14B

facts-first:

```bash
python3 -m src.token_analysis.plot_simple_analysis \
  --input_pkl reports/token_analysis/14b_facts_first_all_hop_raw/patching_results.pkl \
  --summary_json reports/token_analysis/14b_facts_first_all_hop_raw/summary.json \
  --output_dir reports/token_analysis/14b_facts_first_all_hop_raw
```

expr-first:

```bash
python3 -m src.token_analysis.plot_simple_analysis \
  --input_pkl reports/token_analysis/14b_expr_first_all_hop_raw/patching_results.pkl \
  --summary_json reports/token_analysis/14b_expr_first_all_hop_raw/summary.json \
  --output_dir reports/token_analysis/14b_expr_first_all_hop_raw
```

#### Mistral-7B-Instruct-v0.1

facts-first:

```bash
python3 -m src.token_analysis.plot_simple_analysis \
  --input_pkl reports_mistral/token_analysis_facts_first_raw/patching_results.pkl \
  --summary_json reports_mistral/token_analysis_facts_first_raw/summary.json \
  --output_dir reports_mistral/token_analysis_facts_first_raw
```

expr-first:

```bash
python3 -m src.token_analysis.plot_simple_analysis \
  --input_pkl reports_mistral/token_analysis_expr_first_raw/patching_results.pkl \
  --summary_json reports_mistral/token_analysis_expr_first_raw/summary.json \
  --output_dir reports_mistral/token_analysis_expr_first_raw
```

#### Llama-3.1-8B-Instruct

facts-first:

```bash
python3 -m src.token_analysis.plot_simple_analysis \
  --input_pkl reports_llama3/token_analysis_facts_first_raw/patching_results.pkl \
  --summary_json reports_llama3/token_analysis_facts_first_raw/summary.json \
  --output_dir reports_llama3/token_analysis_facts_first_raw
```

expr-first:

```bash
python3 -m src.token_analysis.plot_simple_analysis \
  --input_pkl reports_llama3/token_analysis_expr_first_raw/patching_results.pkl \
  --summary_json reports_llama3/token_analysis_expr_first_raw/summary.json \
  --output_dir reports_llama3/token_analysis_expr_first_raw
```

### 2.2 refined 图

命令模板：

```bash
python3 -m src.token_analysis.plot_refined_analysis \
  --sum_stats_json <refined目录里的refined_stats_sum.json> \
  --mean_stats_json <refined目录里的refined_stats_mean.json> \
  --output_dir <refined目录>
```

会生成：

- `refined_by_stage_sum.png`
- `refined_by_stage_mean.png`

#### Qwen3-8B

facts-first:

```bash
python3 -m src.token_analysis.plot_refined_analysis \
  --sum_stats_json reports/token_analysis/8b_facts_first_all_hop_refined/refined_stats_sum.json \
  --mean_stats_json reports/token_analysis/8b_facts_first_all_hop_refined/refined_stats_mean.json \
  --output_dir reports/token_analysis/8b_facts_first_all_hop_refined
```

expr-first:

```bash
python3 -m src.token_analysis.plot_refined_analysis \
  --sum_stats_json reports/token_analysis/8b_expr_first_all_hop_refined/refined_stats_sum.json \
  --mean_stats_json reports/token_analysis/8b_expr_first_all_hop_refined/refined_stats_mean.json \
  --output_dir reports/token_analysis/8b_expr_first_all_hop_refined
```

#### Qwen3-14B

facts-first:

```bash
python3 -m src.token_analysis.plot_refined_analysis \
  --sum_stats_json reports/token_analysis/14b_facts_first_all_hop_refined/refined_stats_sum.json \
  --mean_stats_json reports/token_analysis/14b_facts_first_all_hop_refined/refined_stats_mean.json \
  --output_dir reports/token_analysis/14b_facts_first_all_hop_refined
```

expr-first:

```bash
python3 -m src.token_analysis.plot_refined_analysis \
  --sum_stats_json reports/token_analysis/14b_expr_first_all_hop_refined/refined_stats_sum.json \
  --mean_stats_json reports/token_analysis/14b_expr_first_all_hop_refined/refined_stats_mean.json \
  --output_dir reports/token_analysis/14b_expr_first_all_hop_refined
```

#### Mistral-7B-Instruct-v0.1

facts-first:

```bash
python3 -m src.token_analysis.plot_refined_analysis \
  --sum_stats_json reports_mistral/token_analysis_facts_first_refined/refined_stats_sum.json \
  --mean_stats_json reports_mistral/token_analysis_facts_first_refined/refined_stats_mean.json \
  --output_dir reports_mistral/token_analysis_facts_first_refined
```

expr-first:

```bash
python3 -m src.token_analysis.plot_refined_analysis \
  --sum_stats_json reports_mistral/token_analysis_expr_first_refined/refined_stats_sum.json \
  --mean_stats_json reports_mistral/token_analysis_expr_first_refined/refined_stats_mean.json \
  --output_dir reports_mistral/token_analysis_expr_first_refined
```

#### Llama-3.1-8B-Instruct

facts-first:

```bash
python3 -m src.token_analysis.plot_refined_analysis \
  --sum_stats_json reports_llama3/token_analysis_facts_first_refined/refined_stats_sum.json \
  --mean_stats_json reports_llama3/token_analysis_facts_first_refined/refined_stats_mean.json \
  --output_dir reports_llama3/token_analysis_facts_first_refined
```

expr-first:

```bash
python3 -m src.token_analysis.plot_refined_analysis \
  --sum_stats_json reports_llama3/token_analysis_expr_first_refined/refined_stats_sum.json \
  --mean_stats_json reports_llama3/token_analysis_expr_first_refined/refined_stats_mean.json \
  --output_dir reports_llama3/token_analysis_expr_first_refined
```

---

## 3. Heads analysis 画图命令

Heads analysis 现在拆成三步图：

1. Step1：layer-head role distribution
2. Step2：taxonomy line chart
3. Step3：PD curves

### 3.1 Step1 画图

命令模板：

```bash
python3 -m src.heads_analysis.plot_step1_heads \
  --classify_csv <classify/top_heads_pattern_labels.csv> \
  --summary_json <step1_summary.json> \
  --output_png <classify/layer_head_role_distribution.png>
```

### 3.2 Step2 画图

命令模板：

```bash
python3 -m src.heads_analysis.plot_step2_taxonomy \
  --counts_csv <taxonomy/head_taxonomy_counts.csv> \
  --output_png <taxonomy/head_taxonomy_line_chart.png>
```

### 3.3 Step3 画图

命令模板：

```bash
python3 -m src.heads_analysis.plot_step3_curves \
  --pd_curve_csv <validation/pd_curve_metrics.csv> \
  --output_dir <validation/plots>
```

会生成：

- `pd_abs_ratio_curve.png`
- `pd_signed_ratio_curve.png`
- `dpd_shift_curve.png`

---

## 4. 各模型 heads analysis 的完整画图命令

下面每个实验目录都需要分别执行 Step1 / Step2 / Step3 三条命令。

### 4.1 Qwen3-8B facts-first

```bash
python3 -m src.heads_analysis.plot_step1_heads \
  --classify_csv reports/heads_analysis_qwen3_8b_facts_first/classify/top_heads_pattern_labels.csv \
  --summary_json reports/heads_analysis_qwen3_8b_facts_first/step1_summary.json \
  --output_png reports/heads_analysis_qwen3_8b_facts_first/classify/layer_head_role_distribution.png
```

```bash
python3 -m src.heads_analysis.plot_step2_taxonomy \
  --counts_csv reports/heads_analysis_qwen3_8b_facts_first/taxonomy/head_taxonomy_counts.csv \
  --output_png reports/heads_analysis_qwen3_8b_facts_first/taxonomy/head_taxonomy_line_chart.png
```

```bash
python3 -m src.heads_analysis.plot_step3_curves \
  --pd_curve_csv reports/heads_analysis_qwen3_8b_facts_first/validation/pd_curve_metrics.csv \
  --output_dir reports/heads_analysis_qwen3_8b_facts_first/validation/plots
```

### 4.2 Qwen3-8B expr-first

```bash
python3 -m src.heads_analysis.plot_step1_heads \
  --classify_csv reports/heads_analysis_qwen3_8b_expr_first/classify/top_heads_pattern_labels.csv \
  --summary_json reports/heads_analysis_qwen3_8b_expr_first/step1_summary.json \
  --output_png reports/heads_analysis_qwen3_8b_expr_first/classify/layer_head_role_distribution.png
```

```bash
python3 -m src.heads_analysis.plot_step2_taxonomy \
  --counts_csv reports/heads_analysis_qwen3_8b_expr_first/taxonomy/head_taxonomy_counts.csv \
  --output_png reports/heads_analysis_qwen3_8b_expr_first/taxonomy/head_taxonomy_line_chart.png
```

```bash
python3 -m src.heads_analysis.plot_step3_curves \
  --pd_curve_csv reports/heads_analysis_qwen3_8b_expr_first/validation/pd_curve_metrics.csv \
  --output_dir reports/heads_analysis_qwen3_8b_expr_first/validation/plots
```

### 4.3 Qwen3-14B facts-first

```bash
python3 -m src.heads_analysis.plot_step1_heads \
  --classify_csv reports/heads_analysis_qwen3_14b_facts_first/classify/top_heads_pattern_labels.csv \
  --summary_json reports/heads_analysis_qwen3_14b_facts_first/step1_summary.json \
  --output_png reports/heads_analysis_qwen3_14b_facts_first/classify/layer_head_role_distribution.png
```

```bash
python3 -m src.heads_analysis.plot_step2_taxonomy \
  --counts_csv reports/heads_analysis_qwen3_14b_facts_first/taxonomy/head_taxonomy_counts.csv \
  --output_png reports/heads_analysis_qwen3_14b_facts_first/taxonomy/head_taxonomy_line_chart.png
```

```bash
python3 -m src.heads_analysis.plot_step3_curves \
  --pd_curve_csv reports/heads_analysis_qwen3_14b_facts_first/validation/pd_curve_metrics.csv \
  --output_dir reports/heads_analysis_qwen3_14b_facts_first/validation/plots
```

### 4.4 Qwen3-14B expr-first

```bash
python3 -m src.heads_analysis.plot_step1_heads \
  --classify_csv reports/heads_analysis_qwen3_14b_expr_first/classify/top_heads_pattern_labels.csv \
  --summary_json reports/heads_analysis_qwen3_14b_expr_first/step1_summary.json \
  --output_png reports/heads_analysis_qwen3_14b_expr_first/classify/layer_head_role_distribution.png
```

```bash
python3 -m src.heads_analysis.plot_step2_taxonomy \
  --counts_csv reports/heads_analysis_qwen3_14b_expr_first/taxonomy/head_taxonomy_counts.csv \
  --output_png reports/heads_analysis_qwen3_14b_expr_first/taxonomy/head_taxonomy_line_chart.png
```

```bash
python3 -m src.heads_analysis.plot_step3_curves \
  --pd_curve_csv reports/heads_analysis_qwen3_14b_expr_first/validation/pd_curve_metrics.csv \
  --output_dir reports/heads_analysis_qwen3_14b_expr_first/validation/plots
```

### 4.5 Mistral facts-first

```bash
python3 -m src.heads_analysis.plot_step1_heads \
  --classify_csv reports_mistral/heads_analysis_facts_first/classify/top_heads_pattern_labels.csv \
  --summary_json reports_mistral/heads_analysis_facts_first/step1_summary.json \
  --output_png reports_mistral/heads_analysis_facts_first/classify/layer_head_role_distribution.png
```

```bash
python3 -m src.heads_analysis.plot_step2_taxonomy \
  --counts_csv reports_mistral/heads_analysis_facts_first/taxonomy/head_taxonomy_counts.csv \
  --output_png reports_mistral/heads_analysis_facts_first/taxonomy/head_taxonomy_line_chart.png
```

```bash
python3 -m src.heads_analysis.plot_step3_curves \
  --pd_curve_csv reports_mistral/heads_analysis_facts_first/validation/pd_curve_metrics.csv \
  --output_dir reports_mistral/heads_analysis_facts_first/validation/plots
```

### 4.6 Mistral expr-first

```bash
python3 -m src.heads_analysis.plot_step1_heads \
  --classify_csv reports_mistral/heads_analysis_expr_first/classify/top_heads_pattern_labels.csv \
  --summary_json reports_mistral/heads_analysis_expr_first/step1_summary.json \
  --output_png reports_mistral/heads_analysis_expr_first/classify/layer_head_role_distribution.png
```

```bash
python3 -m src.heads_analysis.plot_step2_taxonomy \
  --counts_csv reports_mistral/heads_analysis_expr_first/taxonomy/head_taxonomy_counts.csv \
  --output_png reports_mistral/heads_analysis_expr_first/taxonomy/head_taxonomy_line_chart.png
```

```bash
python3 -m src.heads_analysis.plot_step3_curves \
  --pd_curve_csv reports_mistral/heads_analysis_expr_first/validation/pd_curve_metrics.csv \
  --output_dir reports_mistral/heads_analysis_expr_first/validation/plots
```

### 4.7 Llama facts-first

```bash
python3 -m src.heads_analysis.plot_step1_heads \
  --classify_csv reports_llama3/heads_analysis_facts_first/classify/top_heads_pattern_labels.csv \
  --summary_json reports_llama3/heads_analysis_facts_first/step1_summary.json \
  --output_png reports_llama3/heads_analysis_facts_first/classify/layer_head_role_distribution.png
```

```bash
python3 -m src.heads_analysis.plot_step2_taxonomy \
  --counts_csv reports_llama3/heads_analysis_facts_first/taxonomy/head_taxonomy_counts.csv \
  --output_png reports_llama3/heads_analysis_facts_first/taxonomy/head_taxonomy_line_chart.png
```

```bash
python3 -m src.heads_analysis.plot_step3_curves \
  --pd_curve_csv reports_llama3/heads_analysis_facts_first/validation/pd_curve_metrics.csv \
  --output_dir reports_llama3/heads_analysis_facts_first/validation/plots
```

### 4.8 Llama expr-first

```bash
python3 -m src.heads_analysis.plot_step1_heads \
  --classify_csv reports_llama3/heads_analysis_expr_first/classify/top_heads_pattern_labels.csv \
  --summary_json reports_llama3/heads_analysis_expr_first/step1_summary.json \
  --output_png reports_llama3/heads_analysis_expr_first/classify/layer_head_role_distribution.png
```

```bash
python3 -m src.heads_analysis.plot_step2_taxonomy \
  --counts_csv reports_llama3/heads_analysis_expr_first/taxonomy/head_taxonomy_counts.csv \
  --output_png reports_llama3/heads_analysis_expr_first/taxonomy/head_taxonomy_line_chart.png
```

```bash
python3 -m src.heads_analysis.plot_step3_curves \
  --pd_curve_csv reports_llama3/heads_analysis_expr_first/validation/pd_curve_metrics.csv \
  --output_dir reports_llama3/heads_analysis_expr_first/validation/plots
```

---

## 5. 不需要单独画图命令的部分

下面这些实验目前不需要额外的“plot 命令”：

- `src.eval.inference`
- `src.eval.filtering`
- `src.eval.metrics`
- 数据集生成

它们本身主要产生：

- jsonl 预测结果
- dual-correct 过滤结果
- json 指标文件

---

## 6. 最推荐的使用方式

如果你现在是按“先算结果、后画图”的新流程跑，推荐顺序是：

1. 先跑 MLP / token / heads 的结果生成
2. 确认对应目录下已有 `csv/json/pkl`
3. 再执行本文里的画图命令
4. 如果想重画图，直接重复执行本文命令即可，不需要重跑模型
