"""Master runner for heads_analysis (low-complexity standalone pipeline)."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.model_loading import add_model_source_arg
from src.progress import log_event, resolve_log_path, setup_file_logger

from .step1_discover_fast import run_step1
from .step2_taxonomy import run_step2
from .step3_validate_fast import run_step3


def main() -> None:
    parser = argparse.ArgumentParser(description="heads_analysis: low-complexity 3-step pipeline")
    parser.add_argument("--model_id", required=True)
    add_model_source_arg(parser)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--input", default="")
    parser.add_argument("--hop", default="one_hop", choices=["one_hop", "two_hop", "all"])
    parser.add_argument("--prompt_order", default="facts_first")
    parser.add_argument("--prompt_style", default="symbolic")

    # Step1 fast
    parser.add_argument("--impact_samples", type=int, default=500, help="Balanced per-rule budget for Step1 refine stage.")
    parser.add_argument("--probe_samples", type=int, default=64, help="Balanced per-rule budget for Step1 probe stage.")
    parser.add_argument("--classify_samples", type=int, default=500, help="Balanced per-rule budget for Step1 classification stage.")
    parser.add_argument("--top_n", type=int, default=64)
    parser.add_argument("--top_m_per_layer", type=int, default=4)
    parser.add_argument("--candidate_pool_mult", type=int, default=4)
    parser.add_argument("--quantile_keep", type=float, default=0.6)
    parser.add_argument("--late_layer_frac", type=float, default=0.0)
    parser.add_argument("--token_scope", default="all_tokens", choices=["query_only", "all_tokens"])
    parser.add_argument("--score_mode", default="zscore")
    parser.add_argument("--eval_batch_size", type=int, default=8)

    # Step3 fast
    parser.add_argument("--validation_samples", type=int, default=400, help="Balanced per-rule budget for Step3 PD curves.")
    parser.add_argument("--validation_accuracy_samples", type=int, default=500, help="Balanced per-rule budget for Step3 accuracy table.")
    parser.add_argument("--k_values", default="1,2,4,8,16,32,64")
    parser.add_argument("--random_trials_min", type=int, default=6)
    parser.add_argument("--random_trials_max", type=int, default=20)
    parser.add_argument("--random_sem_target", type=float, default=0.01)
    parser.add_argument(
        "--include_signed_ratio_plot",
        "--include-signed-ratio-plot",
        dest="include_signed_ratio_plot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Generate pd_signed_ratio_curve.png in Step3. Disabled by default.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress_every", type=int, default=10)
    parser.add_argument("--steps", default="1,2,3")
    parser.add_argument(
        "--save_plots",
        "--save-plots",
        dest="save_plots",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    steps = {int(s.strip()) for s in args.steps.split(",") if s.strip()}
    invalid_steps = sorted(steps.difference({1, 2, 3}))
    if invalid_steps:
        raise ValueError(f"Unsupported steps after removing Step4: {invalid_steps}. Use steps from {{1,2,3}}.")
    out = args.output_dir
    logger = setup_file_logger(__name__, resolve_log_path(output_dir=out, filename="run_all.log"))

    classify_json = out / "classify" / "top_heads_pattern_labels.json"
    classify_csv = out / "classify" / "top_heads_pattern_labels.csv"
    n_layers = 0

    if 1 in steps:
        log_event(logger, "=" * 60)
        log_event(logger, "HEADS_ANALYSIS STEP 1 (FAST): impact + classification")
        log_event(logger, "=" * 60)
        s1 = run_step1(
            model_id=args.model_id,
            output_dir=out,
            input_path=args.input,
            hop=args.hop,
            prompt_order=args.prompt_order,
            prompt_style=args.prompt_style,
            impact_samples=args.impact_samples,
            probe_samples=args.probe_samples,
            classify_samples=args.classify_samples,
            top_n=args.top_n,
            top_m_per_layer=args.top_m_per_layer,
            candidate_pool_mult=args.candidate_pool_mult,
            quantile_keep=args.quantile_keep,
            late_layer_frac=args.late_layer_frac,
            token_scope=args.token_scope,
            score_mode=args.score_mode,
            eval_batch_size=args.eval_batch_size,
            seed=args.seed,
            device=args.device,
            model_source=args.model_source,
            progress_every=args.progress_every,
            save_plots=args.save_plots,
            include_signed_ratio_plot=args.include_signed_ratio_plot,
        )
        n_layers = int(s1.get("n_layers", 0))

    if 2 in steps:
        log_event(logger, "=" * 60)
        log_event(logger, "HEADS_ANALYSIS STEP 2: taxonomy")
        log_event(logger, "=" * 60)
        if not classify_csv.exists():
            raise FileNotFoundError(f"Step1 output not found: {classify_csv}")
        run_step2(
            classify_csv=classify_csv,
            output_dir=out / "taxonomy",
            n_layers=n_layers,
            role_col="role_label",
            save_plots=args.save_plots,
        )

    if 3 in steps:
        log_event(logger, "=" * 60)
        log_event(logger, "HEADS_ANALYSIS STEP 3 (FAST): role validation")
        log_event(logger, "=" * 60)
        if not classify_json.exists():
            raise FileNotFoundError(f"Step1 output not found: {classify_json}")
        run_step3(
            model_id=args.model_id,
            classify_json=classify_json,
            output_dir=out / "validation",
            input_path=args.input,
            hop=args.hop,
            prompt_order=args.prompt_order,
            prompt_style=args.prompt_style,
            max_samples=args.validation_samples,
            accuracy_samples=args.validation_accuracy_samples,
            k_values=args.k_values,
            late_layer_frac=args.late_layer_frac,
            token_scope=args.token_scope,
            random_trials_min=args.random_trials_min,
            random_trials_max=args.random_trials_max,
            random_sem_target=args.random_sem_target,
            eval_batch_size=args.eval_batch_size,
            seed=args.seed,
            device=args.device,
            model_source=args.model_source,
            progress_every=args.progress_every,
            save_plots=args.save_plots,
        )

    log_event(logger, "=" * 60)
    log_event(logger, "HEADS_ANALYSIS COMPLETE")
    log_event(logger, "=" * 60)
    log_event(logger, f"Output directory: {out}")
    if args.save_plots:
        log_event(logger, f"Step1 classify plot: {out}/classify/layer_head_role_distribution.png")
        log_event(logger, f"Step2 taxonomy plot: {out}/taxonomy/head_taxonomy_line_chart.png")
        log_event(logger, f"Step3 plots dir: {out}/validation/plots")
    else:
        log_event(logger, "Plots skipped. Use the dedicated plot_* modules to render from saved csv/json files.")


if __name__ == "__main__":
    main()
