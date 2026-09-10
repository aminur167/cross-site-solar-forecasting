"""
Master Pipeline Orchestrator
Executes the complete scientific research workflow end-to-end:
Data Prep -> Baselines -> Cross-Site -> Uncertainty -> Deep Learning -> SHAP -> Figures -> Unit Tests.
Author: Aminur Rahman (HSTU) | Project: Cross-Site Solar PV Power Forecasting
"""

import os
import sys
import time
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def run_step(step_name: str, script_path: str):
    print("\n" + "#"*80)
    print(f"STEP: {step_name.upper()}")
    print(f"Executing: {script_path}")
    print("#"*80)
    start = time.time()
    result = subprocess.run([sys.executable, script_path], check=True)
    elapsed = time.time() - start
    print(f">> SUCCESS: {step_name} completed in {elapsed:.1f}s.")

def main():
    print("\n" + "="*80)
    print("STARTING END-TO-END RESEARCH WORKFLOW EXECUTION")
    print("Repository: https://github.com/aminur167/cross-site-solar-forecasting")
    print("="*80)

    total_start = time.time()

    steps = [
        ("Data Engineering & Preprocessing", "src/data/prepare_datasets.py"),
        ("Baseline Benchmark Evaluation", "experiments/run_baseline_benchmark.py"),
        ("Cross-Site Transfer & Few-Shot Adaptation", "experiments/run_cross_site_experiments.py"),
        ("Conformal Uncertainty & Sensor Robustness", "experiments/run_uncertainty_and_robustness.py"),
        ("Deep Learning (LSTM/TCN) & DM Statistics", "experiments/run_deep_learning_benchmark.py"),
        ("SHAP Explainability & Attribution", "src/evaluation/explainability.py"),
        ("Figure 1 Generation", "src/visualization/plot_benchmark.py"),
        ("Figure 2 Generation", "src/visualization/plot_cross_site_adaptation.py"),
        ("Figure 3 Generation", "src/visualization/plot_uncertainty_robustness.py"),
        ("Unit Tests & Methodological Integrity", "-m unittest tests/test_pipeline.py")
    ]

    for name, path in steps:
        if path.startswith("-m"):
            cmd = [sys.executable] + path.split()
            subprocess.run(cmd, check=True)
        else:
            run_step(name, path)

    total_elapsed = time.time() - total_start
    print("\n" + "="*80)
    print(f"ALL EXPERIMENTS AND CHECKS COMPLETED SUCCESSFULLY IN {total_elapsed:.1f}s!")
    print("Results and figures are updated in experiments/results/, figures/, and tables/.")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
