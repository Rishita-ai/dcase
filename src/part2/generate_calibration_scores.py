from pathlib import Path
import sys
import csv
import numpy as np

PROJECT = Path(__file__).resolve().parent
HANDOVER = PROJECT / "DCASE_2026_PART1_COMPLETE" / "Handover"
DATASET = Path.home() / "Downloads" / "DCASE2026_AllMachines"

OUTPUT = PROJECT / "Part2_Results" / "Calibration_Scores"
OUTPUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(HANDOVER / "Preprocessing"))
import dcase_part1_lib as p1

MACHINES = {
    "ToyCar": "ToyCar",
    "ToyCarEmu": "ToyCarEmu",
    "Fan": "fan",
    "GearboxEmu": "gearboxEmu",
    "BearingEmu": "bearingEmu",
    "SlideRailEmu": "sliderEmu",
    "ValveEmu": "valveEmu"
}

VARIANTS = [
    ("Model_A", "Source_Only", "source_only"),
    ("Model_B", "Source_Target", "source_target")
]

COLUMNS = [
    "machine", "model_variant", "calibration_role",
    "statistics_reference", "filename", "domain",
    "number_of_windows", "mse_score",
    "latent_mahalanobis_score",
    "residual_mahalanobis_score"
]


def score_recording(wav, model, ck, stats):

    x = p1.wav_to_model_input(wav, ck)
    output = p1.run_inference(model, x)

    mse = float(np.mean(output["window_mse"]))

    residual_frames = output["residual"].reshape(
        -1, ck["frame_dim"]
    )

    latent_distance = p1.mahalanobis_sq(
        output["latent"],
        stats["latent_mean"],
        stats["latent_precision"]
    )

    residual_distance = p1.mahalanobis_sq(
        residual_frames,
        stats["residual_mean"],
        stats["residual_precision"]
    )

    values = [
        mse,
        float(np.mean(latent_distance)),
        float(np.mean(residual_distance))
    ]

    if not np.isfinite(values).all():
        raise ValueError(f"Invalid score: {wav.name}")

    return values, len(output["window_mse"])


def main():

    total_rows = 0

    for machine, dataset_folder in MACHINES.items():

        print(f"\n========== {machine} ==========")

        train_folder = DATASET / dataset_folder / "train"
        all_files = sorted(train_folder.glob("*.wav"))

        for model_name, model_folder, prefix in VARIANTS:

            print(f"\nScoring {model_name} training recordings...")

            model_dir = (
                HANDOVER / "Models" / machine / model_folder
            )

            model, ck = p1.load_checkpoint(
                model_dir / f"{prefix}_autoencoder.pth"
            )

            if model_name == "Model_A":

                source_files = [
                    f for f in all_files
                    if "_source_train_normal_" in f.name
                ]

                target_files = [
                    f for f in all_files
                    if "_target_train_normal_" in f.name
                ]

                if len(source_files) != 990 or len(target_files) != 10:
                    raise ValueError("Model A training counts incorrect")

                groups = [
                    ("source_train_calibration",
                     source_files, "source_train"),

                    ("target_train_sanity",
                     target_files, "source_train")
                ]

            else:

                all_normal_files = [
                    f for f in all_files
                    if "_train_normal_" in f.name
                ]

                if len(all_normal_files) != 1000:
                    raise ValueError("Model B training count incorrect")

                groups = [
                    ("all_train_calibration",
                     all_normal_files, "all_train")
                ]

            rows = []

            for role, files, reference_group in groups:

                stats_path = (
                    HANDOVER / "Mahalanobis_Statistics" / machine
                    / f"{prefix}_{reference_group}_mahalanobis_stats.npz"
                )

                stats = p1.load_mahalanobis_stats(stats_path)

                for i, wav in enumerate(files):

                    scores, windows = score_recording(
                        wav, model, ck, stats
                    )

                    domain = (
                        "source" if "_source_" in wav.name
                        else "target"
                    )

                    rows.append({
                        "machine": machine,
                        "model_variant": model_name,
                        "calibration_role": role,
                        "statistics_reference": reference_group,
                        "filename": wav.name,
                        "domain": domain,
                        "number_of_windows": windows,
                        "mse_score": scores[0],
                        "latent_mahalanobis_score": scores[1],
                        "residual_mahalanobis_score": scores[2]
                    })

                    if (i + 1) % 100 == 0:
                        print(f"{role}: {i + 1}/{len(files)}")

                print(f"{role}: completed {len(files)} recordings")

            output_file = (
                OUTPUT / f"{machine}_{model_name}_calibration.csv"
            )

            with open(output_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=COLUMNS)
                writer.writeheader()
                writer.writerows(rows)

            total_rows += len(rows)

            print("Saved:", output_file)
            print("Rows:", len(rows))

            del model

    print("\nALL CALIBRATION SCORING COMPLETED!")
    print("Total calibration and sanity-check rows:", total_rows)


if __name__ == "__main__":
    main()
