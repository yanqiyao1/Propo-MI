python -m src.heads_analysis.run_all \
    --model_id Qwen/Qwen3-14B \
    --model_source huggingface \
    --input artifacts/filtered_dual_correct_14b.jsonl \
    --output_dir reports/heads_analysis_14b_one_hop_top512 \
    --hop one_hop \
    --prompt_order facts_first \
    --prompt_style symbolic \
    --impact_samples 2000 \
    --probe_samples 256 \
    --classify_samples 2000 \
    --top_n 512 \
    --top_m_per_layer 8 \
    --candidate_pool_mult 1 \
    --quantile_keep 0.6 \
    --late_layer_frac 0.0 \
    --token_scope query_only \
    --score_mode zscore \
    --eval_batch_size 8 \
    --validation_samples 1500 \
    --k_values 1,2,4,8,16,32,64 \
    --random_trials_min 6 \
    --random_trials_max 20 \
    --random_sem_target 0.01 \
    --seed 42 \
    --device cuda \
    --progress_every 10 \
    --steps 1,2,3

  推荐命令：14B two-hop

  python -m src.heads_analysis.run_all \
    --model_id Qwen/Qwen3-14B \
    --model_source huggingface \
    --input artifacts/filtered_dual_correct_14b.jsonl \
    --output_dir reports/heads_analysis_14b_two_hop_top512 \
    --hop two_hop \
    --prompt_order facts_first \
    --prompt_style symbolic \
    --impact_samples 2000 \
    --probe_samples 256 \
    --classify_samples 2000 \
    --top_n 512 \
    --top_m_per_layer 8 \
    --candidate_pool_mult 1 \
    --quantile_keep 0.6 \
    --late_layer_frac 0.0 \
    --token_scope query_only \
    --score_mode zscore \
    --eval_batch_size 8 \
    --validation_samples 1500 \
    --k_values 1,2,4,8,16,32,64 \
    --random_trials_min 6 \
    --random_trials_max 20 \
    --random_sem_target 0.01 \
    --seed 42 \
    --device cuda \
    --progress_every 10 \
    --steps 1,2,3

  如果你显存或时间比较紧
  可以先用一个“轻量版 top512”：

  - probe_samples=128
  - impact_samples=1000
  - classify_samples=1000
  - validation_samples=800
  - eval_batch_size=4

  轻量版命令

  python -m src.heads_analysis.run_all \
    --model_id Qwen/Qwen3-14B \
    --model_source huggingface \
    --input artifacts/filtered_dual_correct_14b.jsonl \
    --output_dir reports/heads_analysis_14b_one_hop_top512_light_v2 \
    --hop one_hop \
    --prompt_order facts_first \
    --prompt_style symbolic \
    --impact_samples 1000 \
    --probe_samples 128 \
    --classify_samples 1000 \
    --top_n 512 \
    --top_m_per_layer 8 \
    --candidate_pool_mult 1 \
    --quantile_keep 0.6 \
    --late_layer_frac 0.0 \
    --token_scope query_only \
    --score_mode zscore \
    --eval_batch_size 4 \
    --validation_samples 800 \
    --k_values 1,2,4,8,16,32 \
    --random_trials_min 6 \
    --random_trials_max 20 \
    --random_sem_target 0.01 \
    --seed 42 \
    --device cuda \
    --progress_every 10 \
    --steps 1,2,3
