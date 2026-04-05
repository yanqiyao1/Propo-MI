 src/token_analysis 里真正的实验入口只有两个：
      - src/token_analysis/activation_patching_dataset.py
      - src/token_analysis/refined_token_analysis.py
  - 另外两个文件是辅助模块，不是单独 CLI：
      - src/token_analysis/refined_token_classifier.py
      - src/token_analysis/refined_plotting.py

  这个实验怎么跑

  - 第一步：activation_patching_dataset
      - 对每个样本做 token-position × layer 的 residual activation patching
      - patch 逻辑在 src/token_analysis/activation_patching_dataset.py
      - 它会输出 patching_results.pkl、simple 分类统计和基础图，供 refined token analysis 继续使用
  - 第二步：refined_token_analysis
      - 读取上一步的 patching_results.pkl
      - 用更细的 token 分类器重新聚合并画 refined 图
      - 入口在 src/token_analysis/refined_token_analysis.py

  重要提醒

  - 这个模块里的 max_samples 目前还是顺序截断，不是按规则均匀抽样
  - 位置在 src/token_analysis/activation_patching_dataset.py
  - 所以如果你设 --max_samples 1000，它会取过滤后的前 1000 条

  推荐输入

  - 如果你要做稳定的 patching 分析，建议直接用：
      - artifacts/filtered_dual_correct_14b.jsonl
      - 或 artifacts/filtered_dual_correct_8b.jsonl
  - 这种情况下建议显式加 --require_dual_correct

  ———

  推荐完整命令：14B one-hop

  - 第一步：生成 patching 数据与基础图

  python -m src.token_analysis.activation_patching_dataset \
    --model_id Qwen/Qwen3-14B \
    --model_source huggingface \
    --input artifacts/filtered_dual_correct_14b.jsonl \
    --output_dir reports/token_analysis/14b_one_hop_raw \
    --prompt_style symbolic \
    --hop one_hop \
    --max_samples 1000 \
    --require_dual_correct \
    --strict_length_match \
    --early_end 14 \
    --middle_end 24 \
    --device cuda:1 \
    --progress_every 10 \
    --save_plots

  - 第二步：做 refined token analysis

  python -m src.token_analysis.refined_token_analysis \
    --input_pkl reports/token_analysis/14b_one_hop_raw/patching_results.pkl \
    --output_dir reports/token_analysis/14b_one_hop_refined \
    --title "Qwen3-14B One-hop Refined Token Analysis" \
    --early_end 14 \
    --middle_end 24 \
    --n_layers 40 \
    --include-derived-assignment \
    --save-plots \
    --save-csv

  ———

  推荐完整命令：14B two-hop

  - 第一步：生成 patching 数据与基础图

  python -m src.token_analysis.activation_patching_dataset \
    --model_id Qwen/Qwen3-14B \
    --model_source huggingface \
    --input artifacts/filtered_dual_correct_14b.jsonl \
    --output_dir reports/token_analysis/14b_two_hop_raw \
    --prompt_style symbolic \
    --hop two_hop \
    --max_samples 1000 \
    --require_dual_correct \
    --strict_length_match \
    --early_end 14 \
    --middle_end 24 \
    --device cuda \
    --progress_every 10 \
    --save_plots

  - 第二步：做 refined token analysis

  python -m src.token_analysis.refined_token_analysis \
    --input_pkl reports/token_analysis/14b_two_hop_raw/patching_results.pkl \
    --output_dir reports/token_analysis/14b_two_hop_refined \
    --title "Qwen3-14B Two-hop Refined Token Analysis" \
    --early_end 14 \
    --middle_end 24 \
    --n_layers 40 \
    --include-derived-assignment \
    --save-plots \
    --save-csv

  ———

  如果你想直接跑 all-hop

  - 第一步

  python -m src.token_analysis.activation_patching_dataset \
    --model_id Qwen/Qwen3-14B \
    --model_source huggingface \
    --input artifacts/filtered_dual_correct_14b.jsonl \
    --output_dir reports/token_analysis/14b_all_hop_raw \
    --prompt_style symbolic \
    --hop all \
    --max_samples 1000 \
    --require_dual_correct \
    --strict_length_match \
    --early_end 14 \
    --middle_end 24 \
    --device cuda \
    --progress_every 10 \
    --save_plots

  - 第二步

  python -m src.token_analysis.refined_token_analysis \
    --input_pkl reports/token_analysis/14b_all_hop_raw/patching_results.pkl \
    --output_dir reports/token_analysis/14b_all_hop_refined \
    --title "Qwen3-14B All-hop Refined Token Analysis" \
    --early_end 14 \
    --middle_end 24 \
    --n_layers 40 \
    --include-derived-assignment \
    --save-plots \
    --save-csv

  ———

  如果你想跑 8B
  把下面两处替换即可：

  - Qwen/Qwen3-14B → Qwen/Qwen3-8B
  - artifacts/filtered_dual_correct_14b.jsonl → artifacts/filtered_dual_correct_8b.jsonl

  另外 refined 阶段的层数改成：

  - --n_layers 36

  例如：

  python -m src.token_analysis.activation_patching_dataset \
    --model_id Qwen/Qwen3-8B \
    --model_source huggingface \
    --input artifacts/filtered_dual_correct_8b.jsonl \
    --output_dir reports/token_analysis/8b_one_hop_raw \
    --prompt_style symbolic \
    --hop one_hop \
    --max_samples 1000 \
    --require_dual_correct \
    --strict_length_match \
    --early_end 14 \
    --middle_end 24 \
    --device cuda \
    --progress_every 10 \
    --save_plots

  python -m src.token_analysis.refined_token_analysis \
    --input_pkl reports/token_analysis/8b_one_hop_raw/patching_results.pkl \
    --output_dir reports/token_analysis/8b_one_hop_refined \
    --title "Qwen3-8B One-hop Refined Token Analysis" \
    --early_end 14 \
    --middle_end 24 \
    --n_layers 36 \
    --include-derived-assignment \
    --save-plots \
    --save-csv

  ———

  输出文件

  - 第一步 activation_patching_dataset 会产出：
      - patching_results.pkl
      - failed_samples.json
      - statistics_simple.csv
      - category_comparison_simple.png
      - layer_stage_simple.png
      - heatmap_simple.png
      - summary.json
      - 写出逻辑在 src/token_analysis/activation_patching_dataset.py
  - 第二步 refined_token_analysis 会产出：
      - refined_stats_sum.json
      - refined_stats_mean.json
      - refined_metadata.json
      - refined_stats_sum.csv
      - refined_stats_mean.csv
      - refined_by_stage_sum.png
      - refined_by_stage_mean.png
      - 写出逻辑在 src/token_analysis/refined_token_analysis.py
