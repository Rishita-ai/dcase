from pathlib import Path
import sys
import numpy as np

PROJECT = Path.home() / "Downloads" / "DCASE2026_Part2_Work"
HANDOVER = PROJECT / "DCASE_2026_PART1_COMPLETE" / "Handover"
DATASET = Path.home() / "Downloads" / "DCASE2026_AllMachines"

sys.path.insert(0, str(HANDOVER / "Preprocessing"))

import dcase_part1_lib as p1

# One real normal ToyCar test recording
test_folder = DATASET / "ToyCar" / "test"

wav_files = sorted(test_folder.glob("*source_test_normal*.wav"))

if not wav_files:
    raise FileNotFoundError("No source normal ToyCar test WAV found")

wav_path = wav_files[0]

print("Test recording:", wav_path.name)

variants = {
    "Model A": "Source_Only",
    "Model B": "Source_Target"
}

for model_name, folder_name in variants.items():

    folder = HANDOVER / "Models" / "ToyCar" / folder_name

    prefix = "source_only" if model_name == "Model A" else "source_target"

    checkpoint = folder / f"{prefix}_autoencoder.pth"

    model, ck = p1.load_checkpoint(checkpoint)

    x = p1.wav_to_model_input(wav_path, ck)

    output = p1.run_inference(model, x)

    stats_path = (
        HANDOVER / "Mahalanobis_Statistics" / "ToyCar"
        / f"{prefix}_source_train_mahalanobis_stats.npz"
    )

    stats = p1.load_mahalanobis_stats(stats_path)

    latent_score = p1.mahalanobis_sq(
        output["latent"],
        stats["latent_mean"],
        stats["latent_precision"]
    )

    residual_frames = output["residual"].reshape(
        -1, ck["frame_dim"]
    )

    residual_score = p1.mahalanobis_sq(
        residual_frames,
        stats["residual_mean"],
        stats["residual_precision"]
    )

    print(f"\n{model_name}")
    print("Input shape:", x.shape)
    print("Reconstruction shape:", output["reconstruction"].shape)
    print("Mean MSE:", round(float(output["window_mse"].mean()), 6))
    print("Mean latent Mahalanobis:", round(float(latent_score.mean()), 6))
    print("Mean residual Mahalanobis:", round(float(residual_score.mean()), 6))

    assert np.isfinite(output["window_mse"]).all()
    assert np.isfinite(latent_score).all()
    assert np.isfinite(residual_score).all()

print("\nBOTH MODEL TESTS PASSED!")
