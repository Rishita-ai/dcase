# Models, statistics and data: what is and is not in this repository

| Item | Size | In repo? | Where to get it |
|---|---|---|---|
| Raw dataset (7 ZIP archives) | about 4.8 GB | No | Zenodo record 19336329 (DCASE 2026 Task 2 development set) |
| 14 trained checkpoints (`*_autoencoder.pth`) | 1.8 MB each, 25 MB total | No (`.gitignore`) | `DCASE_2026_PART1_COMPLETE_HANDOVER.zip` (55 MB) attached as a GitHub Release asset or shared from Google Drive |
| Epoch snapshots (`epoch_NNN.pth`, `*_last.pth`) | about 340 MB | No | not needed for inference; keep privately |
| Mahalanobis statistics (`*.npz`, 42 files) | 0.8 MB each, 33 MB | No | same handover ZIP |
| Per-recording score, calibration and prediction CSVs | about 3 MB | No (kept with the results package) | `Part2_Results.zip` as a Release asset |
| Metrics, thresholds, comparisons, figures, histories | small | Yes | `results/` |
| Preprocessing/model library `dcase_part1_lib.py` | small | Yes | `src/` |
| Audit, verification and protocol reports | small | Yes | `docs/` |

To run inference or the app, download the handover ZIP, extract it so that `DCASE_2026_PART1_COMPLETE/Handover/` sits next to `app.py`, and place `Part2_Results/Thresholds/all_machine_thresholds.csv` as the scripts expect (see README).

Nothing in this repository contains credentials or personal data. The notebook mounts the author's own Google Drive paths; adjust `CONFIG` for your account.
