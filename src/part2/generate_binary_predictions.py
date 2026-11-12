import csv
from pathlib import Path

ROOT = Path(__file__).parent
TEST_DIR = ROOT / "Part2_Results" / "Test_Scores"
THRESHOLD_FILE = ROOT / "Part2_Results" / "Thresholds" / "all_machine_thresholds.csv"
OUTPUT_DIR = ROOT / "Part2_Results" / "Binary_Predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(THRESHOLD_FILE, newline="") as f:
    thresholds = list(csv.DictReader(f))

threshold_lookup = {
    (r["machine"], r["model_variant"], r["scoring_method"]): r
    for r in thresholds
}

all_predictions = []

for file in TEST_DIR.glob("*_scores.csv"):
    if file.name == "ALL_MACHINES_ALL_MODELS_scores.csv":
        continue

    with open(file, newline="") as f:
        recordings = list(csv.DictReader(f))

    predictions = []

    for recording in recordings:
        machine = recording["machine"]
        model = recording["model_variant"]

        for method in ["MSE", "Latent_Mahalanobis", "Residual_Mahalanobis"]:
            threshold_row = threshold_lookup[(machine, model, method)]
            reference = threshold_row["statistics_reference"]

            if method == "MSE":
                score_column = "mse_score"
            elif method == "Latent_Mahalanobis":
                score_column = f"latent_mahalanobis_{reference}"
            else:
                score_column = f"residual_mahalanobis_{reference}"

            score = float(recording[score_column])
            threshold = float(threshold_row["threshold"])

            predictions.append({
                "machine": machine,
                "model_variant": model,
                "filename": recording["filename"],
                "domain": recording["domain"],
                "scoring_method": method,
                "score": score,
                "threshold": threshold,
                "prediction": "ABNORMAL" if score >= threshold else "NORMAL"
            })

    output_file = OUTPUT_DIR / file.name.replace("_scores.csv", "_predictions.csv")

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=predictions[0].keys())
        writer.writeheader()
        writer.writerows(predictions)

    all_predictions.extend(predictions)
    print(f"Created: {output_file.name} ({len(predictions)} predictions)")

combined_file = OUTPUT_DIR / "ALL_MACHINES_ALL_MODELS_predictions.csv"

with open(combined_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=all_predictions[0].keys())
    writer.writeheader()
    writer.writerows(all_predictions)

print(f"\nTOTAL PREDICTIONS: {len(all_predictions)}")
print(f"Saved in: {OUTPUT_DIR}")
