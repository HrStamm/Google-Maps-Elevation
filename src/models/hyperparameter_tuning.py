import os
import sys
import pandas as pd
import numpy as np

# Ensure Python can find the 'src' module when running this script directly
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.models.train_model import BayesianOptimizationSearch

def run_tuning_experiment():
    kappas = [0.5, 1.0, 1.5, 2.0, 2.5]
    variances = [1.0, 2.0]
    n_seeds = 5
    n_iterations = 50   # Overriding to make sure it matches user's request
    
    results = []
    
    tuning_file = os.path.join(PROJECT_ROOT, "reports", "hyperparameter_tuning_results.csv")
    os.makedirs(os.path.dirname(tuning_file), exist_ok=True)
    
    total_combinations = len(kappas) * len(variances) * n_seeds
    current_run = 0
    
    print(f"Starting Hyperparameter Tuning...")
    print(f"Testing {len(kappas)} kappas x {len(variances)} variances = {len(kappas)*len(variances)} combinations.")
    print(f"Running {n_seeds} times per combination for average (Total Runs: {total_combinations})")
    
    for kappa in kappas:
        for kernel_var in variances:
            print(f"\n" + "="*50)
            print(f"Testing combination: kappa={kappa}, kernel_variance={kernel_var}")
            print("="*50)
            
            for seed in range(n_seeds):
                current_run += 1
                print(f"\n--- Run {current_run}/{total_combinations} (seed={seed}) ---")
                
                # Initialize BO model
                bo = BayesianOptimizationSearch()
                # Override hyperparameters
                bo.kappa = kappa
                bo.kernel_variance = kernel_var
                bo.n_iterations = n_iterations
                
                # Suppress the massive print spam per run by capturing standard output or let it run
                # Running search
                bo_results = bo.run_search(seed=seed)
                
                best_temp = bo_results['best_temperature']
                
                # Save the results for this exact run
                results.append({
                    "kappa": kappa,
                    "kernel_variance": kernel_var,
                    "seed": seed,
                    "best_temperature": best_temp
                })
                
                # Update the CSV proactively
                df = pd.DataFrame(results)
                df.to_csv(tuning_file, index=False)
                
    # Analysis Phase
    print("\n\n" + "="*50)
    print("HYPERPARAMETER TUNING SUMMARY")
    print("="*50)
    df = pd.DataFrame(results)
    
    # Calculate average best temperature for each hyperparameter combination
    summary = df.groupby(['kappa', 'kernel_variance']).agg(
        avg_best_temp=('best_temperature', 'mean'),
        std_best_temp=('best_temperature', 'std'),
        max_best_temp=('best_temperature', 'max'),
        min_best_temp=('best_temperature', 'min'),
        runs=('best_temperature', 'count')
    ).reset_index()
    
    # Sort by the highest average best temperature
    summary = summary.sort_values(by='avg_best_temp', ascending=False)
    
    print("\nPerformance for each combination (Ranked by Average Best Temp):")
    print(summary.to_string(index=False))
    
    best_combo = summary.iloc[0]
    print("\n🏆 BEST PARAMETERS FOUND 🏆")
    print(f"Kappa: {best_combo['kappa']}")
    print(f"Kernel Variance: {best_combo['kernel_variance']}")
    print(f"Average Best Temp across {n_seeds} runs: {best_combo['avg_best_temp']:.2f}°C")
    
    # Save summary
    summary_file = os.path.join(PROJECT_ROOT, "reports", "hyperparameter_tuning_summary.csv")
    summary.to_csv(summary_file, index=False)
    print(f"\n[+] Full raw results saved to: {tuning_file}")
    print(f"[+] Summary report saved to: {summary_file}")

if __name__ == "__main__":
    run_tuning_experiment()
