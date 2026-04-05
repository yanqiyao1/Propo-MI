set -e

plot_heads_experiment() {
  local output_dir="$1"

  python3 -m src.heads_analysis.plot_step1_heads \
    --classify_csv "${output_dir}/classify/top_heads_pattern_labels.csv" \
    --summary_json "${output_dir}/step1_summary.json" \
    --output_png "${output_dir}/classify/layer_head_role_distribution.png"

  python3 -m src.heads_analysis.plot_step2_taxonomy \
    --counts_csv "${output_dir}/taxonomy/head_taxonomy_counts.csv" \
    --output_png "${output_dir}/taxonomy/head_taxonomy_line_chart.png"

  python3 -m src.heads_analysis.plot_step3_curves \
    --pd_curve_csv "${output_dir}/validation/pd_curve_metrics.csv" \
    --output_dir "${output_dir}/validation/plots"
}

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
  --eval_batch_size 8 \
  --seed 42 \
  --device cuda:0 \
  --steps 1,2,3 \
  --no-save-plots

plot_heads_experiment "reports/heads_analysis_qwen3_14b_facts_first"

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
  --eval_batch_size 8 \
  --seed 42 \
  --device cuda:0 \
  --steps 1,2,3 \
  --no-save-plots

plot_heads_experiment "reports/heads_analysis_qwen3_14b_expr_first"

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
  --top_n 512\
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
  --device cuda:0 \
  --steps 1,2,3 \
  --no-save-plots

plot_heads_experiment "reports/heads_analysis_qwen3_8b_facts_first"

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
  --eval_batch_size 8 \
  --seed 42 \
  --device cuda:0 \
  --steps 1,2,3 \
  --no-save-plots

plot_heads_experiment "reports/heads_analysis_qwen3_8b_expr_first"