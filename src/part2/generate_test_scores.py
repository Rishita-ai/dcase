from pathlib import Path
import sys
import csv
import numpy as np

PROJECT = Path(__file__).resolve().parent
HANDOVER = PROJECT / "DCASE_2026_PART1_COMPLETE" / "Handover"
DATASET = Path.home() / "Downloads" / "DCASE2026_AllMachines"

OUTPUT = PROJECT / "Part2_Results" / "Test_Scores"
OUTPUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(HANDOVER / "Preprocessing"))

import dcase_part1_lib as p1


MACHINE_MAP = {
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

GROUPS = ["source_train", "target_train", "all_train"]

COLUMNS = [
    "machine",
    "model_variant",
    "domain",
    "filename",
    "number_of_windows",
    "mse_score"
]

for group in GROUPS:
    COLUMNS.append(f"latent_mahalanobis_{group}")
    COLUMNS.append(f"residual_mahalanobis_{group}")


def get_domain(filename):

    if "_source_" in filename:
        return "source"

    if "_target_" in filename:
        return "target"

    raise ValueError("Unknown test domain: " + filename)


def main():

    all_rows = []

    for machine, folder_name in MACHINE_MAP.items():

        print(f"\n========== {machine} ==========")

        test_folder = DATASET / folder_name / "test"
        test_files = sorted(test_folder.glob("*.wav"))

        if len(test_files) != 200:
            raise ValueError(
                f"{machine}: expected 200 test files, found {len(test_files)}"
            )

        for model_name, model_folder, prefix in VARIANTS:

            print(f"\nProcessing {model_name}...")

            model_dir = (
                HANDOVER / "Models" / machine / model_folder
            )

            checkpoint = (
                model_dir / f"{prefix}_autoencoder.pth"
            )

            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)

            model, ck = p1.load_checkpoint(str(checkpoint))

            if ck["input_dim"] != 1280:
                raise ValueError(
                    f"Unexpected input dimension for {machine}"
                )

            # Load all three normal-reference statistics groups.
            reference_stats = {}

            for group in GROUPS:

                stats_path = (
                    HANDOVER / "Mahalanobis_Statistics" / machine
                    / f"{prefix}_{group}_mahalanobis_stats.npz"
                )

                if not stats_path.exists():
                    raise FileNotFoundError(stats_path)

                reference_stats[group] = (
                    p1.load_mahalanobis_stats(stats_path)
                )

            rows = []

            for index, wav_path in enumerate(test_files):

                domain = get_domain(wav_path.name)

                # Exact Part 1 preprocessing.
                x = p1.wav_to_model_input(wav_path, ck)

                # Autoencoder inference.
                output = p1.run_inference(model, x)

                # One continuous MSE score per recording.
                mse_score = float(
                    np.mean(output["window_mse"])
                )

                # Residual features: one 256-D vector per frame.
                residual_frames = output["residual"].reshape(
                    -1, ck["frame_dim"]
                )

                row = {
                    "machine": machine,
                    "model_variant": model_name,
                    "domain": domain,
                    "filename": wav_path.name,
                    "number_of_windows": len(output["window_mse"]),
                    "mse_score": mse_score
                }

                # Calculate latent and residual distances
                # against source, target and all-training statistics.
                for group in GROUPS:

                    stats = reference_stats[group]

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

                    if not (
                        np.isfinite(latent_distance).all()
                        and np.isfinite(residual_distance).all()
                    ):
                        raise ValueError(
                            f"Invalid Mahalanobis scores: {wav_path.name}"
                        )

                    row[f"latent_mahalanobis_{group}"] = float(
                        np.mean(latent_distance)
                    )

                    row[f"residual_mahalanobis_{group}"] = float(
                        np.mean(residual_distance)
                    )

                if not np.isfinite(mse_score):
                    raise ValueError(
                        f"Invalid MSE score: {wav_path.name}"
                    )

                rows.append(row)
                all_rows.append(row)

                if (index + 1) % 25 == 0:
                    print(
                        f"{model_name}: {index + 1}/200 recordings scored"
                    )

            # Save each machine and model separately.
            output_file = (
                OUTPUT / f"{machine}_{model_name}_scores.csv"
            )

            with open(
                output_file, "w", newline="", encoding="utf-8"
            ) as file:

                writer = csv.DictWriter(
                    file, fieldnames=COLUMNS
                )

                writer.writeheader()
                writer.writerows(rows)

            print("Saved:", output_file)
            print("Recordings scored:", len(rows))

            del model, reference_stats

    # Save combined results.
    combined_file = OUTPUT / "ALL_MACHINES_ALL_MODELS_scores.csv"

    with open(
        combined_file, "w", newline="", encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file, fieldnames=COLUMNS
        )

        writer.writeheader()
        writer.writerows(all_rows)

    print("\n======================================")
    print("ALL SCORING COMPLETED SUCCESSFULLY!")
    print("Total evaluations:", len(all_rows))
    print("Combined CSV:", combined_file)
    print("======================================")


if __name__ == "__main__":
    main()
