import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
METRICS = ROOT / "Part2_Results" / "Metrics"
OUTPUT = METRICS / "Graphs"
OUTPUT.mkdir(parents=True, exist_ok=True)

with open(METRICS / "all_machine_metrics.csv", newline="") as f:
    data = list(csv.DictReader(f))

with open(METRICS / "Comparisons" / "average_comparison.csv", newline="") as f:
    average = list(csv.DictReader(f))

labels = [
    r["model_variant"].replace("_", " ") + "\n" + r["scoring_method"].replace("_", " ")
    for r in average
]
x = np.arange(len(labels))

# Graph 1: Average AUC
values = [float(r["average_AUC_all"]) for r in average]
plt.figure(figsize=(10, 5))
plt.bar(x, values)
plt.xticks(x, labels)
plt.ylim(0, 1)
plt.ylabel("AUC")
plt.title("Average AUC Across 7 Machine Types")
plt.tight_layout()
plt.savefig(OUTPUT / "average_auc_comparison.png", dpi=200)
plt.close()

# Graph 2: Average pAUC
values = [float(r["average_pAUC_max_FPR_0.1"]) for r in average]
plt.figure(figsize=(10, 5))
plt.bar(x, values)
plt.xticks(x, labels)
plt.ylim(0, 1)
plt.ylabel("pAUC")
plt.title("Average pAUC Across 7 Machine Types")
plt.tight_layout()
plt.savefig(OUTPUT / "average_pauc_comparison.png", dpi=200)
plt.close()

# Graph 3: Average source and target F1
source = [float(r["average_F1_source"]) for r in average]
target = [float(r["average_F1_target"]) for r in average]
plt.figure(figsize=(11, 5))
plt.bar(x - 0.18, source, width=0.36, label="Source F1")
plt.bar(x + 0.18, target, width=0.36, label="Target F1")
plt.xticks(x, labels)
plt.ylim(0, 1)
plt.ylabel("F1 Score")
plt.title("Average F1 Score Across 7 Machine Types")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT / "average_f1_comparison.png", dpi=200)
plt.close()

# Graphs 4-6: AUC by machine, comparing Model A and Model B
machines = sorted(set(r["machine"] for r in data))
methods = ["MSE", "Latent_Mahalanobis", "Residual_Mahalanobis"]

for method in methods:
    model_a = []
    model_b = []

    for machine in machines:
        a = next(r for r in data if r["machine"] == machine
                 and r["model_variant"] == "Model_A"
                 and r["scoring_method"] == method)
        b = next(r for r in data if r["machine"] == machine
                 and r["model_variant"] == "Model_B"
                 and r["scoring_method"] == method)

        model_a.append(float(a["AUC_all"]))
        model_b.append(float(b["AUC_all"]))

    positions = np.arange(len(machines))
    plt.figure(figsize=(10, 5))
    plt.bar(positions - 0.18, model_a, width=0.36, label="Model A")
    plt.bar(positions + 0.18, model_b, width=0.36, label="Model B")
    plt.xticks(positions, machines, rotation=25)
    plt.ylim(0, 1)
    plt.ylabel("AUC")
    plt.title(f"{method.replace('_', ' ')}: AUC by Machine")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT / f"machine_auc_{method}.png", dpi=200)
    plt.close()

print("All graphs created successfully!")
print("Total graphs:", len(list(OUTPUT.glob("*.png"))))
print("Saved in:", OUTPUT)
