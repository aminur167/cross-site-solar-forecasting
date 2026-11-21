"""
Master Pipeline Orchestrator
============================
Executes the complete scientific research workflow end-to-end:
Data Prep -> Baselines -> Cross-Site -> Uncertainty -> Deep Learning -> SHAP -> Figures -> Unit Tests.

Author: Aminur Rahman (HSTU) | Project: Cross-Site Solar PV Power Forecasting
"""

import os
import sys
import time
import argparse
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_step(step_name: str, script_path: str):
    print("\n" + "#" * 80)
    print(f"STEP: {step_name.upper()}")
    print(f"Executing: {script_path}")
    print("#" * 80)
    start = time.time()
    repo_root = os.path.dirname(os.path.abspath(__file__))
    subprocess.run([sys.executable, script_path], check=True, cwd=repo_root)
    elapsed = time.time() - start
    print(f">> SUCCESS: {step_name} completed in {elapsed:.1f}s.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-End Solar PV Transfer Learning & Conformal Forecasting Pipeline."
    )
    parser.add_argument(
        "--skip-dl",
        action="store_true",
        help="Skip PyTorch LSTM/TCN training to accelerate CPU execution.",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip automated pipeline integrity unit tests.",
    )
    parser.add_argument(
        "--generate-parity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate Figure 5 parity and residual diagnostic plot.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 80)
    print("STARTING END-TO-END RESEARCH WORKFLOW EXECUTION")
    print("Repository: https://github.com/aminur167/cross-site-solar-forecasting")
    print(f"Options: skip_dl={args.skip_dl}, skip_tests={args.skip_tests}, generate_parity={args.generate_parity}")
    print("=" * 80)

    total_start = time.time()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    steps = [
        ("Data Engineering & Preprocessing", "src/data/prepare_datasets.py"),
        ("Raw Data Quality Audit", "src/data/data_quality_auditor.py"),
        ("Baseline Benchmark Evaluation", "experiments/run_baseline_benchmark.py"),
        ("Cross-Site Transfer & Few-Shot Adaptation", "experiments/run_cross_site_experiments.py"),
        ("Conformal Uncertainty & Sensor Robustness", "experiments/run_uncertainty_and_robustness.py"),
    ]

    if not args.skip_dl:
        steps.append(("Deep Learning (LSTM/TCN) & DM Statistics", "experiments/run_deep_learning_benchmark.py"))

    steps.extend([
        ("SHAP Explainability & Attribution", "src/evaluation/explainability.py"),
        ("Figure 1 Generation", "src/visualization/plot_benchmark.py"),
        ("Figure 2 Generation", "src/visualization/plot_cross_site_adaptation.py"),
        ("Figure 3 Generation", "src/visualization/plot_uncertainty_robustness.py"),
        ("Figure 6 Reliability Diagram", "src/visualization/plot_reliability_diagram.py"),
        ("LaTeX Table Export", "src/evaluation/export_latex_tables.py"),
    ])

    if args.generate_parity:
        steps.append(("Figure 5 Parity Plot Generation", "src/visualization/plot_parity.py"))

    if not args.skip_tests:
        steps.append(("Unit Tests & Methodological Integrity", "-m unittest tests/test_pipeline.py"))

    for name, path in steps:
        if path.startswith("-m"):
            cmd = [sys.executable] + path.split()
            subprocess.run(cmd, check=True)
        else:
            run_step(name, path)

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 80)
    print(f"ALL EXPERIMENTS AND CHECKS COMPLETED SUCCESSFULLY IN {total_elapsed:.1f}s!")
    print("Results and figures are updated in experiments/results/, figures/, and tables/.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
