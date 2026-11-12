import csv
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

ROOT = Path(__file__).parent
PRED_DIR = ROOT / "Part2_Results" / "Binary_Predictions"
OUTPUT_DIR = ROOT / "Part2_Results" / "Metrics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

results = []

for file in PRED_DIR.glob("*_predictions.csv"):
    if file.name.startswith("ALL_"):
        continue

    with open(file, newline="") as f:
        rows = list(csv.DictReader(f))

    for method in ["MSE", "Latent_Mahalanobis", "Residual_Mahalanobis"]:
        selected = [r for r in rows if r["scoring_method"] == method]

        labels = [1 if "anomaly" in r["filename"].lower() else 0 for r in selected]
        scores = [float(r["score"]) for r in selected]
        decisions = [1 if r["prediction"] == "ABNORMAL" else 0 for r in selected]
        domains = [r["domain"] for r in selected]

        auc_all = roc_auc_score(labels, scores)
        pauc = roc_auc_score(labels, scores, max_fpr=0.1)

        result = {
            "machine": selected[0]["machine"],
            "model_variant": selected[0]["model_variant"],
            "scoring_method": method,
            "AUC_all": auc_all,
            "pAUC_max_FPR_0.1": pauc
        }

        for domain in ["source", "target"]:
            domain_mask = [d == domain for d in domains]

            # Official DCASE domain-specific AUC calculation
            auc_mask = [
                domains[i] == domain or labels[i] == 1
                for i in range(len(labels))
            ]

            domain_labels = [labels[i] for i in range(len(labels)) if domain_mask[i]]
            domain_decisions = [decisions[i] for i in range(len(decisions)) if domain_mask[i]]

            result[f"AUC_{domain}"] = roc_auc_score(
                [labels[i] for i in range(len(labels)) if auc_mask[i]],
                [scores[i] for i in range(len(scores)) if auc_mask[i]]
            )

            result[f"Precision_{domain}"] = precision_score(
                domain_labels, domain_decisions, zero_division=0
            )
            result[f"Recall_{domain}"] = recall_score(
                domain_labels, domain_decisions, zero_division=0
            )
            result[f"F1_{domain}"] = f1_score(
                domain_labels, domain_decisions, zero_division=0
            )

        results.append(result)

output_file = OUTPUT_DIR / "all_machine_metrics.csv"

with open(output_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("Updated metrics successfully!")
print("Total result rows:", len(results))
print("Saved to:", output_file)
print("\nFirst 3 results:")
for row in results[:3]:
    print(row)
