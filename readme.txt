Project: Unified Experiments (8 Models) for Sleep-Study Labeling

Layout
------
00_embedding_extraction.py
01_embedding_extraction_sorted.ipynb
02_trajectory_analysis.ipynb
10_phate_feature_extraction.ipynb
11_ehr_feature_extraction.ipynb
12_data_prep.ipynb
20_tda_feature_analysis_ahi.ipynb
30_exp_unified_8models.py
31_plotting_overlays.py

Data Inputs (env-based; set before running)
-------------------------------------------
DATA_DIR   : directory with session files:
             <pid>_embeddings.npy
             <pid>_<label>.npy            # sleep_label, desat_label, eeg_label, apnea_label, hypop_label
             <pid>_ehr_feature.npy
             <pid>_time_feature_normalized.npy
             <pid>_phate_point_feature_normalized.npy
TDA_NPZ    : npz with {X, pid} for 6 TDA features (session-level)
ROOT_OUT   : output root for metrics/curves/summaries
SPLIT_DIR  : directory to store shared train/val/test pid lists

Model Registry (paper-spec)
---------------------------
m0_linear_emb                  : Linear probe on embeddings
m0_1_mlp_emb                   : MLP on embeddings
m1_mlp_emb_ehr                 : MLP on emb + EHR
m1_1_mlp_emb_phate             : MLP on emb + (time + point)     # "phate"
m1_2_mlp_emb_tda               : MLP on emb + TDA (6 feats)
m2_mlp_emb_ehr_phate           : MLP on emb + EHR + (time + point)
m2_1_mlp_emb_ehr_tda           : MLP on emb + EHR + TDA
m3_mlp_emb_ehr_phate_tda       : MLP on emb + EHR + (time + point) + TDA

Labels and Classes
------------------
LABEL_DICT = {
  sleep_label: 5,
  desat_label: 2,
  eeg_label:   2,
  apnea_label: 2,
  hypop_label: 2
}

Training / Evaluation
---------------------
- Shared, stratified session splits persisted to SPLIT_DIR (double-blind).
- Means/stds computed on train only; features standardized and clipped.
- Binary tasks: focal loss (γ=1.5) with class weights; early stop on Val AUPRC.
- Multiclass tasks: weighted cross-entropy; early stop on Val macro-F1.
- Test-time metrics:
  * Binary: Acc, F1, ROC-AUC, AUPRC + confusion matrix; per-model PR curves saved.
  * Multiclass: Acc, Balanced Acc, macro-F1 + confusion matrix.
- PR/ROC overlays (identical axes) collected from per-model .npz curves.

How to Run
----------
# Example (bash):
export DATA_DIR=/path/to/output_embeddings
export TDA_NPZ=/path/to/tda_features_pidkey_6feat.npz
export ROOT_OUT=/path/to/exp_unified_8
export SPLIT_DIR=/path/to/splits_shared_unified

python -u 30_exp_unified_8models.py

Outputs
-------
ROOT_OUT/
  <model_tag>/
    <model_tag>__<label>__metrics.json
    <model_tag>__<label>__PR.png
    <model_tag>__<label>__curves.npz
  overlays/<label>/
    <label>_PR_overlay.png
    <label>_ROC_overlay.png
  summary_all_results.json

Reproducibility
---------------
- Fixed seeds for numpy/torch.
- Persistent splits; reruns reuse exact session partitions.
- Environment-variable paths avoid institution/user leakage.
