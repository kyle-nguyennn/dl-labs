"""Hyperparameter sweep for VAE KNN accuracy."""
import subprocess, yaml, copy, os, sys

PYTHON = "/home/dnguyen3/miniforge3/envs/cs7643-a4/bin/python"
BASE_DIR = "/home/dnguyen3/code/dl-labs/assignment4"
CONFIG_PATH = os.path.join(BASE_DIR, "configs/config_vae_knn.yaml")

# Base config
with open(CONFIG_PATH) as f:
    base = yaml.safe_load(f)

# Sweep 2: higher beta + latent_dim 16/32 (round 1 showed l64 collapses, low beta hurts)
experiments = [
    # Higher betas with latent_dim=32
    {"name": "b0.02_l32",   "beta": 0.02,  "latent_dim": 32, "hidden_dim": 512, "lr": 0.001, "epochs": 50},
    {"name": "b0.05_l32",   "beta": 0.05,  "latent_dim": 32, "hidden_dim": 512, "lr": 0.001, "epochs": 50},
    {"name": "b0.1_l32",    "beta": 0.1,   "latent_dim": 32, "hidden_dim": 512, "lr": 0.001, "epochs": 50},
    # Smaller latent dim (more concentrated info per dim)
    {"name": "b0.008_l16",  "beta": 0.008, "latent_dim": 16, "hidden_dim": 512, "lr": 0.001, "epochs": 50},
    {"name": "b0.02_l16",   "beta": 0.02,  "latent_dim": 16, "hidden_dim": 512, "lr": 0.001, "epochs": 50},
    # More epochs with baseline 
    {"name": "b0.008_l32_e100", "beta": 0.008, "latent_dim": 32, "hidden_dim": 512, "lr": 0.001, "epochs": 100},
    # Higher capacity + higher beta
    {"name": "b0.02_l32_h768", "beta": 0.02, "latent_dim": 32, "hidden_dim": 768, "lr": 0.001, "epochs": 50},
    # Moderate beta sweep around baseline
    {"name": "b0.015_l32",  "beta": 0.015, "latent_dim": 32, "hidden_dim": 512, "lr": 0.001, "epochs": 50},
]

results = []

for exp in experiments:
    print(f"\n{'='*60}")
    print(f"Experiment: {exp['name']}")
    print(f"  beta={exp['beta']}, latent_dim={exp['latent_dim']}, hidden_dim={exp['hidden_dim']}, lr={exp['lr']}, epochs={exp['epochs']}")
    print(f"{'='*60}")

    # Write config
    cfg = copy.deepcopy(base)
    cfg['vae']['beta'] = exp['beta']
    cfg['network']['latent_dim'] = exp['latent_dim']
    cfg['network']['hidden_dim'] = exp['hidden_dim']
    cfg['train']['lr'] = exp['lr']
    cfg['train']['n_epochs'] = exp['epochs']

    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    output_dir = f"outputs/knn_sweep/{exp['name']}"

    # Train
    train_cmd = [PYTHON, "train.py", "--config_file", CONFIG_PATH, "--output_dir", output_dir]
    r = subprocess.run(train_cmd, cwd=BASE_DIR, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  TRAIN FAILED: {r.stderr[-500:]}")
        results.append((exp['name'], "FAILED", 0))
        continue

    # Eval KNN
    eval_cmd = [PYTHON, "eval_knn.py", "--config_file", CONFIG_PATH, "--output_dir", output_dir, "--n_shots", "10", "--n_test", "10000"]
    r = subprocess.run(eval_cmd, cwd=BASE_DIR, capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print(f"  EVAL FAILED: {r.stderr[-500:]}")
        results.append((exp['name'], "FAILED", 0))
        continue

    # Parse accuracy for k=3
    for line in r.stdout.split('\n'):
        if 'k=3' in line and 'Full' not in line:
            acc = float(line.strip().split('= ')[1])
            results.append((exp['name'], "OK", acc))
            break

print(f"\n{'='*60}")
print("SWEEP RESULTS (10-shot KNN, k=3)")
print(f"{'='*60}")
print(f"{'Experiment':<25} {'Status':<10} {'Accuracy':<10}")
print("-" * 45)
for name, status, acc in results:
    print(f"{name:<25} {status:<10} {acc:.4f}")

# Find best
best = max(results, key=lambda x: x[2])
print(f"\nBest: {best[0]} with accuracy {best[2]:.4f}")
