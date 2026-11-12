import csv
from pathlib import Path

root = Path(__file__).parent
folder = root / "Part2_Results" / "Metrics"
input_file = folder / "all_machine_metrics.csv"
output_folder = folder / "Comparisons"
output_folder.mkdir(exist_ok=True)

with open(input_file, newline="") as f:
    data = list(csv.DictReader(f))

metrics = [
    "AUC_all", "pAUC_max_FPR_0.1",
    "AUC_source", "AUC_target",
    "F1_source", "F1_target"
]

lookup = {
    (r["machine"], r["scoring_method"], r["model_variant"]): r
    for r in data
}

comparisons = []

for machine in sorted(set(r["machine"] for r in data)):
    for method in ["MSE", "Latent_Mahalanobis", "Residual_Mahalanobis"]:
        a = lookup[(machine, method, "Model_A")]
        b = lookup[(machine, method, "Model_B")]

        row = {"machine": machine, "scoring_method": method}

        for metric in metrics:
            av = float(a[metric])
            bv = float(b[metric])
            row[metric + "_Model_A"] = av
            row[metric + "_Model_B"] = bv
            row[metric + "_difference_B_minus_A"] = bv - av

        comparisons.append(row)

with open(output_folder / "model_comparison.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=comparisons[0].keys())
    writer.writeheader()
    writer.writerows(comparisons)

summary = []
for model in ["Model_A", "Model_B"]:
    for method in ["MSE", "Latent_Mahalanobis", "Residual_Mahalanobis"]:
        selected = [
            r for r in data
            if r["model_variant"] == model and r["scoring_method"] == method
        ]

        row = {
            "model_variant": model,
            "scoring_method": method,
            "machines_included": len(selected)
        }

        for metric in metrics:
            row["average_" + metric] = sum(
                float(r[metric]) for r in selected
            ) / len(selected)

        summary.append(row)

with open(output_folder / "average_comparison.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary[0].keys())
    writer.writeheader()
    writer.writerows(summary)

print("Comparison completed!")
print("Per-machine comparisons:", len(comparisons))
print("Average comparison rows:", len(summary))
print("Saved in:", output_folder)
