# Noise-aware Unsupervised Anomalous Sound Detection (DCASE 2026 Task 2)

M.Tech CSE (AI and Data Science) AI Laboratory project. Autoencoder-based anomalous sound detection for seven machine types of the DCASE 2026 Task 2 development dataset, with reconstruction-error (MSE) and Mahalanobis-distance scoring, and a source-only versus source + target training comparison.

| | |
|---|---|
| Part 1 (data audit, preprocessing, model training, handover) | Rishita Vyas (AU2644017) |
| Part 2 (scoring, thresholds, metrics, comparison, Streamlit app) | Tamanna Khirshriya (AU2644031) |


## Results at a glance (test sets, 7 machines x 200 recordings, recomputed in the final audit)

| Model | Scoring | AUC | pAUC (FPR <= 0.1) | AUC source | AUC target |
|---|---|---|---|---|---|
| A (990 source) | MSE | 0.547 | 0.516 | 0.658 | 0.436 |
| A | Latent Mahalanobis | 0.466 | 0.494 | 0.473 | 0.459 |
| A | Residual Mahalanobis | 0.557 | 0.530 | 0.653 | 0.460 |
| B (990 source + 10 target) | MSE | 0.568 | 0.531 | 0.653 | 0.484 |
| B | Latent Mahalanobis | 0.457 | 0.491 | 0.485 | 0.429 |
| B | **Residual Mahalanobis** | **0.573** | **0.537** | 0.650 | 0.496 |

Thresholds are the 90th percentile of a Gamma fit to training-normal scores (official-baseline method). Because no independent validation set exists under the adopted protocol, they are optimistic: false-positive rate on normal test recordings is 0.34 to 0.36 instead of the nominal 0.10. See the report, Chapter 11.

## Repository layout

```
notebooks/   DCASE_2026_PART1_MASTER_COLAB.ipynb (clean) and *_executed.ipynb (with the real run's outputs)
src/         dcase_part1_lib.py (preprocessing, model, inference) and part2/ scripts (scoring, thresholds, metrics, app.py)
results/     metrics, thresholds, comparisons, training histories, figures
docs/        report PDF, slides PDF, Part 1 implementation report, protocol and calibration note, audit and verification reports
part1_handover_small/  configurations and per-machine preprocessing configs (no binaries)
MODELS_AND_DATA.md     what is excluded from the repository and where to get it
```

## Reproducing

### Part 1 (training) in Google Colab
1. Upload the seven dataset ZIPs to `MyDrive/DCASE_2026/Dataset/` with their original names.
2. Open `notebooks/DCASE_2026_PART1_MASTER_COLAB.ipynb` in Colab, choose a GPU runtime, Run all. The notebook audits the archives, trains Model A and Model B per machine (fixed 100 epochs, seed 42), computes Mahalanobis statistics, verifies every model and writes `MyDrive/DCASE_2026_PART1_COMPLETE/` plus a verified handover ZIP. It resumes automatically after a disconnect.

### Part 2 (scoring and evaluation) on a laptop
```
pip install -r requirements.txt
# place the extracted dataset folders (ToyCar, ToyCarEmu, fan, gearboxEmu, bearingEmu, sliderEmu, valveEmu)
# under ~/Downloads/DCASE2026_AllMachines and the handover under ./DCASE_2026_PART1_COMPLETE/Handover
python src/part2/generate_test_scores.py          # 2800 test recordings -> Part2_Results/Test_Scores
python src/part2/generate_calibration_scores.py   # 14000 training-normal scores
python src/part2/fit_thresholds.py                # 42 Gamma thresholds
python src/part2/generate_binary_predictions.py   # 8400 decisions
python src/part2/calculate_metrics.py             # AUC, pAUC, precision, recall, F1
python src/part2/compare_results.py && python src/part2/create_result_graphs.py
```
The dataset location is a constant at the top of each script (`DATASET = Path.home()/"Downloads"/"DCASE2026_AllMachines"`); edit it for another machine.

### Application
```
streamlit run src/part2/app.py
```
Select machine, model and scoring method, upload a WAV; the app reports score, threshold and decision using the same code path as the evaluation.

## Method summary
Both channels -> 128-band log-Mel per channel (n_fft 1024, hop 512) -> 256-dim frame -> per-dimension z-score (training statistics stored in the checkpoint) -> 5-frame windows (1280-dim) -> Autoencoder 1280-128-128-128-128-8-128-128-128-128-1280 (BatchNorm + ReLU) -> scores: mean window MSE, mean latent Mahalanobis distance (8-d), mean residual-frame Mahalanobis distance (256-d) against regularised statistics of the training normals.

## Honest status notes
1. Training executed on a CUDA GPU in Colab on 9 Nov 2026 (device recorded in each checkpoint); the handover was re-assembled in a later CPU session.
2. No validation set exists (all normals train); no early stopping; final-epoch checkpoints.
3. The app's inference path was verified programmatically; interactive testing on real recordings is not part of this repository's evidence.
4. Results are not comparable with the official baseline numbers (two-channel features, normalisation).

## License and citation
Code: MIT (add a LICENSE file before publishing). Dataset: see the DCASE 2026 Task 2 / Zenodo licence terms. Please cite the DCASE 2026 Task 2 description and dataset when using this work.
