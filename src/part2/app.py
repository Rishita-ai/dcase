import csv
import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).parent
HANDOVER = ROOT / "DCASE_2026_PART1_COMPLETE" / "Handover"

PREPROCESSING = HANDOVER / "Preprocessing"
sys.path.insert(0, str(PREPROCESSING))

import dcase_part1_lib as p1

THRESHOLD_FILE = (
    ROOT / "Part2_Results" / "Thresholds" / "all_machine_thresholds.csv"
)

MACHINES = [
    "BearingEmu",
    "Fan",
    "GearboxEmu",
    "SlideRailEmu",
    "ToyCar",
    "ToyCarEmu",
    "ValveEmu"
]

MODELS = {
    "Model A - Source Only": ("Model_A", "Source_Only", "source_only"),
    "Model B - Source + Target": ("Model_B", "Source_Target", "source_target")
}

METHODS = {
    "MSE": "MSE",
    "Latent Mahalanobis": "Latent_Mahalanobis",
    "Residual Mahalanobis": "Residual_Mahalanobis"
}


@st.cache_data
def load_thresholds():
    with open(THRESHOLD_FILE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    return {
        (r["machine"], r["model_variant"], r["scoring_method"]): r
        for r in rows
    }


def analyse_audio(wav_path, machine, model_name, method_name):
    model_key, folder, prefix = MODELS[model_name]

    checkpoint = (
        HANDOVER / "Models" / machine / folder
        / f"{prefix}_autoencoder.pth"
    )

    if not checkpoint.exists():
        raise FileNotFoundError(f"Model not found: {checkpoint}")

    thresholds = load_thresholds()
    threshold_row = thresholds[(machine, model_key, method_name)]

    # Use the exact preprocessing saved with the model.
    output = p1.wav_to_everything(str(wav_path), str(checkpoint))

    # Reconstruction error.
    if method_name == "MSE":
        score = float(np.mean(output["window_mse"]))

    else:
        reference = threshold_row["statistics_reference"]

        stats_path = (
            HANDOVER / "Mahalanobis_Statistics" / machine
            / f"{prefix}_{reference}_mahalanobis_stats.npz"
        )

        if not stats_path.exists():
            raise FileNotFoundError(f"Statistics not found: {stats_path}")

        stats = p1.load_mahalanobis_stats(stats_path)

        if method_name == "Latent_Mahalanobis":
            distances = p1.mahalanobis_sq(
                output["latent"],
                stats["latent_mean"],
                stats["latent_precision"]
            )

        else:
            residual_frames = output["residual"].reshape(
                -1, output["frame_dim"]
            )

            distances = p1.mahalanobis_sq(
                residual_frames,
                stats["residual_mean"],
                stats["residual_precision"]
            )

        score = float(np.mean(distances))

    threshold = float(threshold_row["threshold"])

    if not np.isfinite(score):
        raise ValueError("The calculated anomaly score is invalid.")

    prediction = "ABNORMAL" if score >= threshold else "NORMAL"

    return prediction, score, threshold, len(output["window_mse"])


st.set_page_config(
    page_title="Machine Sound Anomaly Detection",
    page_icon="🔊",
    layout="centered"
)

st.title("Machine Sound Anomaly Detection")
st.write("Upload a machine recording to check for unusual sound behaviour.")

machine = st.selectbox("Select machine type", MACHINES)

model_name = st.selectbox(
    "Select Autoencoder model",
    list(MODELS.keys()),
    index=1
)

method_label = st.selectbox(
    "Select scoring method",
    list(METHODS.keys())
)

uploaded_file = st.file_uploader(
    "Upload machine audio (WAV)",
    type=["wav"]
)

if st.button("Analyse Audio", type="primary", use_container_width=True):

    if uploaded_file is None:
        st.warning("Please upload a WAV audio file first.")

    else:
        temp_path = None

        try:
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ) as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_path = Path(temp_file.name)

            with st.spinner("Analysing machine sound..."):
                result, score, threshold, windows = analyse_audio(
                    temp_path,
                    machine,
                    model_name,
                    METHODS[method_label]
                )

            st.divider()
            st.subheader("Analysis Result")

            if result == "NORMAL":
                st.success("NORMAL - No anomaly detected")
            else:
                st.error("ABNORMAL - Anomaly detected")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Anomaly Score", f"{score:.4f}")

            with col2:
                st.metric("Decision Threshold", f"{threshold:.4f}")

            st.write("**Machine:**", machine)
            st.write("**Model:**", model_name)
            st.write("**Scoring method:**", method_label)
            st.write("**Audio windows analysed:**", windows)

            st.caption(
                "Higher scores indicate more unusual sounds. "
                "Thresholds are based on training-set calibration "
                "and may be optimistic because no held-out "
                "validation recordings were used."
            )

        except Exception as error:
            st.error(f"Analysis failed: {error}")

        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

