from pathlib import Path
import csv
import math

folder = Path("Part2_Results") / "Test_Scores"

machines = [
    "ToyCar", "ToyCarEmu", "Fan", "GearboxEmu",
    "BearingEmu", "SlideRailEmu", "ValveEmu"
]

models = ["Model_A", "Model_B"]

score_columns = [
    "mse_score",
    "latent_mahalanobis_source_train",
    "latent_mahalanobis_target_train",
    "latent_mahalanobis_all_train",
    "residual_mahalanobis_source_train",
    "residual_mahalanobis_target_train",
    "residual_mahalanobis_all_train"
]

total = 0

for machine in machines:
    for model in models:

        path = folder / f"{machine}_{model}_scores.csv"

        with open(path, newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))

        filenames = [r["filename"] for r in rows]

        source = sum("_source_" in f for f in filenames)
        target = sum("_target_" in f for f in filenames)
        normal = sum("_normal_" in f for f in filenames)
        anomaly = sum("_anomaly_" in f for f in filenames)

        invalid = sum(
            not math.isfinite(float(r[col]))
            for r in rows
            for col in score_columns
        )

        duplicates = len(filenames) - len(set(filenames))

        passed = (
            len(rows) == 200
            and source == 100
            and target == 100
            and normal == 100
            and anomaly == 100
            and invalid == 0
            and duplicates == 0
        )

        print(f"\n{machine} - {model}")
        print("Recordings:", len(rows))
        print("Source / Target:", source, "/", target)
        print("Normal / Anomaly:", normal, "/", anomaly)
        print("Invalid scores:", invalid)
        print("Duplicate filenames:", duplicates)
        print("Status:", "PASS" if passed else "FAIL")

        if not passed:
            raise ValueError(f"Verification failed: {machine} {model}")

        total += len(rows)

print("\nTOTAL RECORDING-MODEL EVALUATIONS:", total)
print("ALL SCORE FILES VERIFIED SUCCESSFULLY!")
