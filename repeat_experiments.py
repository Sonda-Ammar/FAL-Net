import argparse
import subprocess
import os
import json
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Run experiments repeatedly for statistical variance.')
    parser.add_argument('--runs', type=int, default=3, help='Number of identical runs')
    args = parser.parse_args()

    results_all_runs = []
    
    # We will invoke the main ablation_runner.py multiple times.
    # To keep the outputs separate, ablation_runner needs an output prefix, but to keep it simple
    # we can just run it, collect full_summary.json, and rename it.
    
    for i in range(args.runs):
        print(f"\n{'='*50}\nStarting Statistical Run {i+1}/{args.runs}\n{'='*50}")
        # Run standard ablation
        subprocess.run(["python", "ablation_runner.py", "--epochs", "50"], check=True) # Reduced epochs for repetition
        
        summary_path = "results_Greenbit/ablation/full_summary.json"
        
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                data = json.load(f)
                results_all_runs.append(data)
                
            # Rename so it's not overwritten
            os.rename(summary_path, f"results_Greenbit/ablation/full_summary_run_{i+1}.json")
            
    # Aggregate stats
    print("\n\n" + "=" * 100)
    print(f"STATISTICAL SUMMARY OVER {args.runs} RUNS (Mean ± Std Dev)")
    print("=" * 100)
    print(f"{'Configuration':<20} | {'APCER':<15} | {'BPCER':<15} | {'ACE':<15} | {'EER':<15}")
    print("-" * 100)
    
    if not results_all_runs:
        print("No results found.")
        return
        
    # Assuming all runs have the same configs in the same order
    num_configs = len(results_all_runs[0])
    
    for c_idx in range(num_configs):
        config_name = results_all_runs[0][c_idx]["name"]
        
        apcers = [r[c_idx].get("apcer", 0) for r in results_all_runs]
        bpcers = [r[c_idx].get("bpcer", 0) for r in results_all_runs]
        aces = [r[c_idx].get("ace", 0) for r in results_all_runs]
        eers = [r[c_idx].get("eer", 0) for r in results_all_runs]
        
        print(f"{config_name:<20} | {np.mean(apcers):.2f}±{np.std(apcers):.2f} | {np.mean(bpcers):.2f}±{np.std(bpcers):.2f} | {np.mean(aces):.2f}±{np.std(aces):.2f} | {np.mean(eers):.2f}±{np.std(eers):.2f}")

    print("=" * 100)

if __name__ == "__main__":
    main()
