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