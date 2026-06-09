# PedSleep-TTA

Code for **"Uncovering Trajectory and Topological Signatures in Multimodal Pediatric Sleep Embeddings"** (ML4H 2025, PMLR 297:1392-1411).

[[Paper]](https://doi.org/10.48550/arXiv.2605.14156)

## Overview

We investigate session-wide diagnostic information in sequences of 30-second pediatric PSG epochs embedded by a multimodal masked autoencoder. We test whether augmenting embeddings with PHATE-derived geometric descriptors, persistent homology summaries, and EHR features yields complementary signals for predicting sleep disorder events. Simple linear and MLP models are used for interpretability.

## Pipeline

| Step | Script | Description |
|------|--------|-------------|
| 0 | `00_embedding_extraction.py` | Extract per-epoch embeddings from the pretrained autoencoder |
| 1 | `01_embedding_extraction_sorted.ipynb` | Sort and organize extracted embeddings by session |
| 2 | `02_trajectory_analysis.ipynb` | Analyze embedding trajectories across sleep epochs |
| 10 | `10_phate_feature_extraction.ipynb` | Extract PHATE coordinates and movement descriptors |
| 11 | `11_ehr_feature_extraction.ipynb` | Extract clinical/EHR features |
| 12 | `12_data_prep.ipynb` | Merge and prepare feature sets for modeling |
| 20 | `20_tda_feature_extraction.ipynb` | Compute persistent homology summaries (TDA features) |
| 21 | `21_tda_feature_analysis_ahi.ipynb` | Analyze topological features stratified by AHI |
| 30 | `30_exp_unified_8models.py` | Run unified experiments (linear + MLP, single/late-fusion) |
| 31 | `31_plotting_overlays.py` | Generate result plots |

Run experiments via `run_unified.sh`.

## Citation

```bibtex
@inproceedings{ye2025pedsleep,
  title={Uncovering Trajectory and Topological Signatures in Multimodal Pediatric Sleep Embeddings},
  author={Ye, Scott and Lee, Harlin},
  booktitle={Proceedings of the Fifth Machine Learning for Health Symposium},
  pages={1392--1411},
  year={2025},
  volume={297},
  series={PMLR}
}
```
