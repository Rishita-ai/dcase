from pathlib import Path
import csv
import numpy as np
from scipy.stats import gamma

PROJECT = Path(__file__).resolve().parent
CAL_DIR = PROJECT / "Part2_Results" / "Calibration_Scores"
OUT_DIR = PROJECT / "Part2_Results" / "Thresholds"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MACHINES = [
    "ToyCar", "ToyCarEmu", "Fan", "GearboxEmu",
    "BearingEmu", "SlideRailEmu", "ValveEmu"
]

MODELS = {
    "Model_A": {
        "role": "source_train_calibration",
        "count": 990,
        "reference": "source_train"
    },
    "Model_B": {
        "role": "all_train_calibration",
        "count": 1000,
        "reference": "all_train"
    }
}

SCORES = {
    "MSE": "mse_score",
    "Latent_Mahalanobis": "latent_mahalanobis_score",
    "Residual_Mahalanobis": "residual_mahalanobis_score"
}

QUANTILE = 0.90
results = []


for machine in MACHINES:
    for model, settings in MODELS.items():

        path = CAL_DIR / f"{machine}_{model}_calibration.csv"

        with open(path, newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))

        selected = [
            row for row in rows
            if row["calibration_role"] == settings["role"]
        ]

        if len(selected) != settings["count"]:
            raise ValueError(
                f"Wrong calibration count: {machine} {model}"
            )

        print(f"\n{machine} - {model}")
        print("Calibration recordings:", len(selected))

        for score_name, column in SCORES.items():

            values = np.array(
                [float(row[column]) for row in selected],
                dtype=np.float64
            )

            if not np.isfinite(values).all():
                raise ValueError(f"Invalid values: {machine} {score_name}")

            if np.any(values <= 0):
                raise ValueError(
                    f"Gamma fitting requires positive values: "
                    f"{machine} {model} {score_name}"
                )

            # Fit Gamma distribution with location fixed at zero.
            shape, loc, scale = gamma.fit(values, floc=0)

            threshold = float(
                gamma.ppf(QUANTILE, shape, loc=loc, scale=scale)
            )

            empirical_threshold = float(
                np.quantile(values, QUANTILE)
            )

            results.append({
                "machine": machine,
                "model_variant": model,
                "scoring_method": score_name,
                "statistics_reference": settings["reference"],
                "calibration_set": settings["role"],
                "calibration_count": len(selected),
                "distribution": "Gamma",
                "gamma_shape": shape,
                "gamma_location": loc,
                "gamma_scale": scale,
                "quantile": QUANTILE,
                "threshold": threshold,
                "empirical_90th_percentile": empirical_threshold,
                "calibration_bias": "Training-set scores; potentially optimistic"
            })

            print(f"{score_name}: {threshold:.6f}")


output = OUT_DIR / "all_machine_thresholds.csv"

with open(output, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\n====================================")
print("GAMMA THRESHOLDS GENERATED!")
print("Total thresholds:", len(results))
print("Saved:", output)
print("====================================")
