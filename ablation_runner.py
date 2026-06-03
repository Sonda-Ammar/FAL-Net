# ablation_runner.py
import subprocess
import os
import json
import argparse


def run_config(name, rank, ssa, multiscale, freeze=False, epochs=100):
    output_dir = f"results_Greenbit/ablation/{name}"

    cmd = [
        "python", "train_ablation.py",
        "--rank", str(rank),
        "--output", output_dir,
        "--epochs", str(epochs)
    ]
    if ssa:
        cmd.append("--ssa")
    else:
        cmd.append("--no-ssa")

    if multiscale:
        cmd.append("--multiscale")
    else:
        cmd.append("--no-multiscale")
        
    if freeze:
        cmd.append("--freeze")

    print(f"\n>>> Running ablation configuration: {name.upper()}")
    print(f">>> command: {' '.join(cmd)}")

    subprocess.run(cmd, check=True)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description='Ablation Study Runner')
    #parser.add_argument('--summary-only', action='store_true', help='Only display the results summary, skip training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
    args = parser.parse_args()

    # Configurations for the paper
    configs = [
        #
        #{"name": "full_finetuning", "rank": 0, "ssa": True, "multiscale": True, "freeze": False},
        #{"name": "frozen_backbone", "rank": 8, "ssa": True, "multiscale": True, "freeze": True},
        #{"name": "baseline", "rank": 0, "ssa": False, "multiscale": False},
        #{"name": "lora_only", "rank": 8, "ssa": False, "multiscale": True},
        #{"name": "ssa_only", "rank": 0, "ssa": True, "multiscale": True},
        #{"name": "proposed_r4", "rank": 4, "ssa": True, "multiscale": True},
        {"name": "proposed_r8", "rank": 8, "ssa": True, "multiscale": True},
        #{"name": "single_scale", "rank": 8, "ssa": True, "multiscale": False},
    ]

    summary = []

    for config in configs:
        output_dir = run_config(
            config["name"],
            config["rank"],
            config["ssa"],
            config["multiscale"],
            freeze=config.get("freeze", False),
            epochs=args.epochs
        )

        # Load results
        res_path = os.path.join(output_dir, "results_metrics.json")
        if os.path.exists(res_path):
            with open(res_path, 'r') as f:
                data = json.load(f)
                summary.append({
                    "name": config["name"],
                    "ace": data.get("ace", 0),
                    "apcer": data.get("apcer", 0),
                    "bpcer": data.get("bpcer", 0),
                    "eer": data.get("eer", 0),
                    "bpcer10": data.get("bpcer10", 0),
                    "bpcer100": data.get("bpcer100", 0),
                    "trainable_params": data.get("trainable_params", 0)
                })

    if not summary:
        print("\n>>> No results found. Please run training first.")
        return

    # Print final Table
    print("\n\n" + "=" * 100)
    print("SUMMARY OF ABLATION RESULTS (Including New Biometric Metrics)")
    print("=" * 100)
    print(f"{'Configuration':<20} | {'APCER':<7} | {'BPCER':<7} | {'ACE':<7} | {'EER':<7} | {'BPCER@10':<9} | {'BPCER@100':<10}")
    print("-" * 100)
    for res in summary:
        print(f"{res['name']:<20} | {res['apcer']:<7.2f} | {res['bpcer']:<7.2f} | {res['ace']:<7.2f} | {res['eer']:<7.2f} | {res['bpcer10']:<9.2f} | {res['bpcer100']:<10.2f}")
    print("=" * 100)

    # Save consolidate summary
    if not os.path.exists("results_Greenbit/ablation"):
        os.makedirs("results_Greenbit/ablation")

    with open("results_Greenbit/ablation/full_summary.json", "w") as f:
        json.dump(summary, f, indent=4)


if __name__ == "__main__":
    main()
