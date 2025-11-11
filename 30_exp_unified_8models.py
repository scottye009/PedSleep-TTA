#!/usr/bin/env python3
"""
Unified training for eight model variants with identical-scale PR/ROC overlays.
"""

# ----------------------------- Imports -----------------------------

import os
import re
import json
from glob import glob
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit

from plotting_overlays import collect_and_plot_overlays, save_curves


# ----------------------------- Config -----------------------------

PAIR_RANGE: slice | None = None
LABEL_DICT: Dict[str, int] = {
    "sleep_label": 5,
    "desat_label": 2,
    "eeg_label": 2,
    "apnea_label": 2,
    "hypop_label": 2,
}

# Double-blind, overridable via environment
DATA_DIR = os.environ.get("DATA_DIR", "/data/output_embeddings")
TDA_NPZ = os.environ.get("TDA_NPZ", "/data/tda/tda_features_pidkey_6feat.npz")
ROOT_OUT = os.environ.get("ROOT_OUT", "/exp/exp_unified_8")
SPLIT_DIR_SHARED = os.environ.get("SPLIT_DIR", "/exp/splits_shared_unified")
os.makedirs(ROOT_OUT, exist_ok=True)
os.makedirs(SPLIT_DIR_SHARED, exist_ok=True)

# Training knobs
SEED = 42
EPOCHS = 50
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 8
CLIP_Z = 8.0
NUM_WORKERS = 0  # NFS-safe
PIN_MEMORY = False

torch.manual_seed(SEED)
np.random.seed(SEED)


# ----------------------------- Helpers: IDs / Paths -----------------------------

def pid_key(pid: str) -> Tuple[int, int, str]:
    """Sort key: (person_id, session_id, pid_str)."""
    nums = re.findall(r"\d+", pid)
    a = int(nums[0]) if len(nums) > 0 else -1
    b = int(nums[1]) if len(nums) > 1 else -1
    return (a, b, pid)


def base_id_from_suffix(path: str, suffix: str) -> str:
    base = os.path.basename(path)
    return base[:-len(suffix)] if base.endswith(suffix) else os.path.splitext(base)[0]


def discover_embedding_pids() -> List[str]:
    files = glob(os.path.join(DATA_DIR, "*_embeddings.npy"))
    ids = [base_id_from_suffix(p, "_embeddings.npy") for p in files]
    ids = sorted(set(ids), key=pid_key)
    return ids[PAIR_RANGE] if isinstance(PAIR_RANGE, slice) else ids


def session_paths(pid: str, label: str) -> Dict[str, str]:
    return {
        "emb": os.path.join(DATA_DIR, f"{pid}_embeddings.npy"),
        "y": os.path.join(DATA_DIR, f"{pid}_{label}.npy"),
        "ehr": os.path.join(DATA_DIR, f"{pid}_ehr_feature.npy"),
        "time": os.path.join(DATA_DIR, f"{pid}_time_feature_normalized.npy"),
        "point": os.path.join(DATA_DIR, f"{pid}_phate_point_feature_normalized.npy"),
    }


def _probe_last_row(arr) -> bool:
    """Light probe to touch the last row; surfaces some broken memmaps early."""
    _ = arr[-1].shape
    return True


# ----------------------------- TDA map -----------------------------

def normalize_pid_key(s: str) -> str:
    nums = re.findall(r"\d+", str(s))
    if len(nums) >= 2:
        return f"{int(nums[0])}_{int(nums[1])}"
    return os.path.splitext(os.path.basename(str(s)))[0]


class TDAMap:
    """Lightweight map pid -> tda feature vector."""

    def __init__(self, npz_path: str):
        self.available = os.path.exists(npz_path) and os.path.isfile(npz_path)
        self.map: Dict[str, np.ndarray] = {}
        self.dim = 0
        if not self.available:
            return
        try:
            pack = np.load(npz_path, allow_pickle=False)
            pids = pack["pid"]
        except Exception:
            pack = np.load(npz_path, allow_pickle=True)
            pids = pack["pid"]
        pids = np.asarray(pids).astype(str)
        X = pack["X"].astype(np.float32)
        for raw_pid, row in zip(pids, X):
            self.map[normalize_pid_key(raw_pid)] = row
        self.dim = X.shape[1]

    def has(self, pid: str) -> bool:
        return normalize_pid_key(pid) in self.map

    def get(self, pid: str) -> np.ndarray:
        return self.map[normalize_pid_key(pid)]


# ----------------------------- Data table / Splits -----------------------------

def build_session_table(
    pids: List[str],
    label: str,
    K: int,
    need: Dict[str, bool],
    tda: TDAMap,
) -> List[Dict]:
    """Collect sessions that have required modalities present."""
    table = []
    for pid in pids:
        p = session_paths(pid, label)
        req = [
            ("emb", True),
            ("y", True),
            ("ehr", need.get("ehr", False)),
            ("time", need.get("time", False)),
            ("point", need.get("point", False)),
        ]
        ok = all((not r) or os.path.exists(p[k]) for k, r in req)
        if not ok:
            continue
        if need.get("tda", False) and not (tda.available and tda.has(pid)):
            continue

        try:
            emb = np.load(p["emb"], mmap_mode="r", allow_pickle=False)
            y = np.load(p["y"], mmap_mode="r", allow_pickle=False)
            _probe_last_row(emb)
            _probe_last_row(y)
            T = int(emb.shape[0])

            if need.get("point", False):
                pt = np.load(p["point"], mmap_mode="r", allow_pickle=False)
                _probe_last_row(pt)
                if pt.shape[0] != T:
                    continue

            if K == 2:
                # Binary stratification: session is positive if ANY epoch > 0
                yb = (y > 0).astype(np.int32)
                strat = int(yb.any())
            else:
                counts = np.bincount(y.astype(np.int64), minlength=K)
                strat = int(np.argmax(counts))

            table.append({"pid": pid, "paths": p, "length": T, "strat": strat})
        except Exception:
            continue

    return table


def _split_files(label: str) -> Tuple[str, str, str]:
    return (
        os.path.join(SPLIT_DIR_SHARED, f"{label}_train.txt"),
        os.path.join(SPLIT_DIR_SHARED, f"{label}_val.txt"),
        os.path.join(SPLIT_DIR_SHARED, f"{label}_test.txt"),
    )


def _save_split(label: str, tr, va, te):
    a, b, c = _split_files(label)
    with open(a, "w") as f:
        f.write("\n".join(tr))
    with open(b, "w") as f:
        f.write("\n".join(va))
    with open(c, "w") as f:
        f.write("\n".join(te))


def _load_split(label: str):
    a, b, c = _split_files(label)
    if not (os.path.exists(a) and os.path.exists(b) and os.path.exists(c)):
        return [], [], []
    rd = lambda p: [x.strip() for x in open(p) if x.strip()]
    return rd(a), rd(b), rd(c)


def split_sessions(
    table,
    label,
    test_size: float = 0.20,
    val_size: float = 0.125,
    seed: int = SEED,
):
    """Stratified 70/10/20 split via (20% test) and (12.5% of remaining for val)."""
    tr, va, te = _load_split(label)
    have = len(tr) > 0 and len(va) > 0 and len(te) > 0

    pids = np.array([r["pid"] for r in table])
    ys = np.array([r["strat"] for r in table])

    if not have:
        sss1 = StratifiedShuffleSplit(1, test_size=test_size, random_state=seed)
        trval_idx, te_idx = next(sss1.split(pids, ys))
        p_trval, p_te = pids[trval_idx], pids[te_idx]
        ys_trval = ys[trval_idx]

        sss2 = StratifiedShuffleSplit(1, test_size=val_size, random_state=seed)
        tr_idx, va_idx = next(sss2.split(p_trval, ys_trval))
        tr = list(map(str, p_trval[tr_idx]))
        va = list(map(str, p_trval[va_idx]))
        te = list(map(str, p_te))
        _save_split(label, tr, va, te)

    S = lambda ids: [r for r in table if r["pid"] in set(ids)]
    return S(tr), S(va), S(te)


# ----------------------------- Stats / Normalization -----------------------------

def _finite_sum_and_count(arr: np.ndarray):
    finite = np.isfinite(arr)
    arr0 = np.where(finite, arr, 0.0)
    return arr0.sum(0, dtype=np.float64), finite.sum(0, dtype=np.int64)


def compute_memmap_mean_std_safe(sessions, key: str):
    s_sum = s_sumsq = s_count = None
    for s in sessions:
        A = np.load(s["paths"][key], mmap_mode="r", allow_pickle=False)
        ss, cc = _finite_sum_and_count(A)
        ss2, cc2 = _finite_sum_and_count(np.square(A, dtype=np.float64))
        if s_sum is None:
            s_sum, s_sumsq, s_count = ss, ss2, cc
        else:
            s_sum += ss
            s_sumsq += ss2
            s_count += cc
    count = np.maximum(s_count, 1)
    mean = s_sum / count
    var = np.maximum(0.0, (s_sumsq / count) - mean**2)
    std = np.sqrt(var)
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def compute_sessionvec_mean_std_safe(sessions, key: str):
    vecs = []
    for s in sessions:
        v = np.load(s["paths"][key], allow_pickle=False).astype(np.float64)
        v[~np.isfinite(v)] = 0.0
        vecs.append(v)
    M = np.vstack(vecs)
    mean = M.mean(0).astype(np.float32)
    std = M.std(0).astype(np.float32)
    std[std < 1e-8] = 1.0
    return mean, std


def compute_tda_mean_std_safe(train_tab, tda_map: TDAMap):
    vecs = [tda_map.get(s["pid"]).astype(np.float64) for s in train_tab]
    for v in vecs:
        v[~np.isfinite(v)] = 0.0
    M = np.vstack(vecs)
    mean = M.mean(0).astype(np.float32)
    std = M.std(0).astype(np.float32)
    std[std < 1e-8] = 1.0
    return mean, std


def standardize_and_clip(x, mean, std, clip=CLIP_Z):
    x = (x - mean) / std
    return np.clip(x, -clip, clip)


def class_counts_from_sessions(sessions, K, binarize):
    c = np.zeros(K, dtype=np.int64)
    for s in sessions:
        y = np.load(s["paths"]["y"], mmap_mode="r", allow_pickle=False)
        if binarize:
            y = (y > 0).astype(np.int64)
            b = np.bincount(y, minlength=2)
        else:
            b = np.bincount(y.astype(np.int64), minlength=K)
        c[: len(b)] += b
    return c


def make_weights(counts, K, device):
    N = counts.sum()
    counts = np.clip(counts, 1, None)
    w = N / (K * counts.astype(np.float32))
    return torch.tensor(w, dtype=torch.float32, device=device)


# ----------------------------- Dataset -----------------------------

class FusionDS(Dataset):
    """
    emb(t) always used.
    Optional branches (by 'use'): ehr(session), time(session), point(t), tda(session).
    phate = time + point (both must be toggled together in specs).
    """

    def __init__(self, sessions, binarize, use: Dict[str, bool], norms, tda_map: TDAMap | None = None):
        self.sessions = sessions
        self.binarize = binarize
        self.use = use
        self.norms = norms
        self.tda_map = tda_map
        self.lengths = [int(s["length"]) for s in sessions]
        self.cum = np.cumsum([0] + self.lengths)
        self.total = int(self.cum[-1])
        self._open = [None] * len(sessions)
        self._dims = None
        self._infer_dims()

    def _infer_dims(self):
        for s in self.sessions:
            p = s["paths"]
            dims = {}
            emb = np.load(p["emb"], mmap_mode="r", allow_pickle=False)
            dims["emb"] = int(emb.shape[1])
            if self.use.get("point"):
                dims["point"] = int(np.load(p["point"], mmap_mode="r", allow_pickle=False).shape[1])
            if self.use.get("ehr"):
                dims["ehr"] = int(np.load(p["ehr"], allow_pickle=False).shape[0])
            if self.use.get("time"):
                dims["time"] = int(np.load(p["time"], allow_pickle=False).shape[0])
            if self.use.get("tda") and self.tda_map is not None:
                dims["tda"] = int(self.tda_map.get(s["pid"]).shape[0])
            self._dims = dims
            return
        raise RuntimeError("no sessions")

    @property
    def feature_dims(self):
        return self._dims

    def __len__(self):
        return self.total

    def _ensure(self, j):
        if self._open[j] is None:
            p = self.sessions[j]["paths"]
            pid = self.sessions[j]["pid"]
            cache = {
                "emb": np.load(p["emb"], mmap_mode="r", allow_pickle=False),
                "y": np.load(p["y"], mmap_mode="r", allow_pickle=False),
            }
            if self.use.get("point"):
                cache["point"] = np.load(p["point"], mmap_mode="r", allow_pickle=False)
            if self.use.get("ehr"):
                v = np.load(p["ehr"], allow_pickle=False).astype(np.float32)
                v[~np.isfinite(v)] = 0.0
                cache["ehr"] = standardize_and_clip(v, self.norms["ehr_mean"], self.norms["ehr_std"]).astype(np.float32)
            if self.use.get("time"):
                v = np.load(p["time"], allow_pickle=False).astype(np.float32)
                v[~np.isfinite(v)] = 0.0
                cache["time"] = standardize_and_clip(v, self.norms["time_mean"], self.norms["time_std"]).astype(np.float32)
            if self.use.get("tda") and self.tda_map is not None:
                v = self.tda_map.get(pid).astype(np.float32)
                v[~np.isfinite(v)] = 0.0
                cache["tda"] = standardize_and_clip(v, self.norms["tda_mean"], self.norms["tda_std"]).astype(np.float32)
            self._open[j] = cache

    def __getitem__(self, idx):
        j = int(np.searchsorted(self.cum, idx, side="right") - 1)
        i = idx - self.cum[j]
        self._ensure(j)
        A = self._open[j]

        x_emb = np.nan_to_num(np.asarray(A["emb"][i], dtype=np.float32).copy(), nan=0.0, posinf=0.0, neginf=0.0)
        x_emb = standardize_and_clip(x_emb, self.norms["emb_mean"], self.norms["emb_std"])

        feats = [torch.from_numpy(x_emb)]
        if self.use.get("ehr"):
            feats.append(torch.from_numpy(A["ehr"]))
        if self.use.get("time"):
            feats.append(torch.from_numpy(A["time"]))
        if self.use.get("point"):
            row = np.nan_to_num(np.asarray(A["point"][i], dtype=np.float32).copy(), nan=0.0, posinf=0.0, neginf=0.0)
            feats.append(torch.from_numpy(standardize_and_clip(row, self.norms["point_mean"], self.norms["point_std"])))
        if self.use.get("tda"):
            feats.append(torch.from_numpy(A["tda"]))

        y = int(A["y"][i])
        if self.binarize:
            y = 1 if y > 0 else 0
        return feats, torch.tensor(y, dtype=torch.long)


# ----------------------------- Models -----------------------------

class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, K: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, K)

    def forward(self, x):
        return self.fc(x)


class FeatureEncoderLN(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class MultiBranchMLP(nn.Module):
    def __init__(self, dims: Dict[str, int], K: int):
        super().__init__()
        encs = []
        if "emb" in dims:
            self.emb = FeatureEncoderLN(dims["emb"])
            encs.append("emb")
        if "ehr" in dims:
            self.ehr = FeatureEncoderLN(dims["ehr"])
            encs.append("ehr")
        if "time" in dims:
            self.time = FeatureEncoderLN(dims["time"])
            encs.append("time")
        if "point" in dims:
            self.point = FeatureEncoderLN(dims["point"])
            encs.append("point")
        if "tda" in dims:
            self.tda = FeatureEncoderLN(dims["tda"])
            encs.append("tda")

        self.branches = encs
        self.classifier = nn.Sequential(
            nn.Linear(128 * len(encs), 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(256, K),
        )

    def forward(self, *xs):
        zs = []
        for name, x in zip(self.branches, xs):
            z = getattr(self, name)(x)
            zs.append(z)
        return self.classifier(torch.cat(zs, dim=1))


# ----------------------------- Registry (8 models) -----------------------------

# phate == time + point
MODEL_SPECS: Dict[str, Dict] = {
    "m0_linear_emb": {"arch": "linear", "use": {"emb": True}},
    "m0_1_mlp_emb": {"arch": "mlp", "use": {"emb": True}},
    "m1_mlp_emb_ehr": {"arch": "mlp", "use": {"emb": True, "ehr": True}},
    "m1_1_mlp_emb_phate": {"arch": "mlp", "use": {"emb": True, "time": True, "point": True}},
    "m1_2_mlp_emb_tda": {"arch": "mlp", "use": {"emb": True, "tda": True}},
    "m2_mlp_emb_ehr_phate": {"arch": "mlp", "use": {"emb": True, "ehr": True, "time": True, "point": True}},
    "m2_1_mlp_emb_ehr_tda": {"arch": "mlp", "use": {"emb": True, "ehr": True, "tda": True}},
    "m3_mlp_emb_ehr_phate_tda": {
        "arch": "mlp",
        "use": {"emb": True, "ehr": True, "time": True, "point": True, "tda": True},
    },
}


# ----------------------------- Losses -----------------------------

class FocalLoss(nn.Module):
    """Focal loss for imbalanced binary tasks (extends to multi-K via one-hot)."""

    def __init__(self, weight=None, gamma: float = 1.5, reduction: str = "mean"):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()
        t = F.one_hot(target, num_classes=logits.size(1)).float()

        if self.weight is not None:
            w = self.weight[target]
        else:
            w = torch.ones_like(target, dtype=logits.dtype, device=logits.device)

        focal = -(1 - (p * t).sum(1)) ** self.gamma * (t * logp).sum(1)
        loss = w * focal

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# ----------------------------- Train / Eval -----------------------------

def nan_guard(*tensors):
    return [
        torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0).clamp_(-CLIP_Z, CLIP_Z)
        for t in tensors
    ]


def run_one(label: str, K: int, model_tag: str, device):
    spec = MODEL_SPECS[model_tag]
    use = spec["use"]
    arch = spec["arch"]
    need = {k: bool(v) for k, v in use.items()}

    # Build table
    all_pids = discover_embedding_pids()
    tda_map = TDAMap(TDA_NPZ) if need.get("tda") else TDAMap("")
    table = build_session_table(all_pids, label, K, need, tda_map)
    if not table:
        print(f"[{model_tag}] no valid sessions for {label}")
        return None

    # Splits
    train_tab, val_tab, test_tab = split_sessions(table, label)
    binarize = K == 2

    # Train-only norms
    emb_mean, emb_std = compute_memmap_mean_std_safe(train_tab, "emb")
    norms = {"emb_mean": emb_mean, "emb_std": emb_std}
    if need.get("point"):
        pm, ps = compute_memmap_mean_std_safe(train_tab, "point")
        norms["point_mean"], norms["point_std"] = pm, ps
    if need.get("ehr"):
        em, es = compute_sessionvec_mean_std_safe(train_tab, "ehr")
        norms["ehr_mean"], norms["ehr_std"] = em, es
    if need.get("time"):
        tm, ts = compute_sessionvec_mean_std_safe(train_tab, "time")
        norms["time_mean"], norms["time_std"] = tm, ts
    if need.get("tda"):
        tdm, tds = compute_tda_mean_std_safe(train_tab, tda_map)
        norms["tda_mean"], norms["tda_std"] = tdm, tds

    # Datasets / loaders
    ds_tr = FusionDS(train_tab, binarize, use, norms, tda_map if need.get("tda") else None)
    ds_va = FusionDS(val_tab, binarize, use, norms, tda_map if need.get("tda") else None)
    ds_te = FusionDS(test_tab, binarize, use, norms, tda_map if need.get("tda") else None)

    dims = ds_tr.feature_dims
    outdir = os.path.join(ROOT_OUT, model_tag)
    os.makedirs(outdir, exist_ok=True)

    dlkw = dict(
        batch_size=BATCH_SIZE,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=False,
    )
    dl_tr = DataLoader(ds_tr, shuffle=True, **dlkw)
    dl_va = DataLoader(ds_va, shuffle=False, **dlkw)
    dl_te = DataLoader(ds_te, shuffle=False, **dlkw)

    # Model
    if arch == "linear" and list(dims.keys()) == ["emb"]:
        model = LinearProbe(dims["emb"], K).to(device)
    else:
        model = MultiBranchMLP(dims, K).to(device)

    # Loss / Opt
    counts = class_counts_from_sessions(train_tab, K, binarize)
    weights = make_weights(counts, K, device)
    criterion = FocalLoss(weight=weights, gamma=1.5) if K == 2 else nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    scaler = GradScaler("cuda", enabled=torch.cuda.is_available())

    # Train loop
    best = -1.0
    best_t = 0.5
    best_state = None
    wait = 0

    for ep in range(1, EPOCHS + 1):
        model.train()
        run = 0.0

        for feats, yb in dl_tr:
            feats = [t.to(device) for t in nan_guard(*feats)]
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=torch.cuda.is_available()):
                logits = model(*feats) if len(feats) > 1 else model(feats[0])
                loss = criterion(logits, yb)

            if not torch.isfinite(loss):
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            run += loss.item() * yb.size(0)

        # Validation
        model.eval()
        P, Y = [], []
        with torch.no_grad():
            for feats, yb in dl_va:
                feats = [t.to(device) for t in nan_guard(*feats)]
                with autocast("cuda", enabled=torch.cuda.is_available()):
                    logits = model(*feats) if len(feats) > 1 else model(feats[0])
                    prob = torch.softmax(logits, dim=1)
                P.append(prob.cpu())
                Y.append(yb)

        P = torch.cat(P).numpy() if P else np.zeros((0, K), dtype=float)
        Y = torch.cat(Y).numpy() if Y else np.zeros((0,), dtype=int)

        if Y.size == 0:
            metric = 0.0
            cur_t = best_t
        else:
            if binarize:
                ap = average_precision_score(Y, P[:, 1])
                thr = np.linspace(0, 1, 101)
                f1s = [f1_score(Y, (P[:, 1] >= t).astype(int)) for t in thr]
                cur_t = float(thr[int(np.argmax(f1s))])
                metric = ap
            else:
                yhat = np.argmax(P, 1)
                metric = f1_score(Y, yhat, average="macro")
                cur_t = best_t

        scheduler.step(metric)

        if metric > best:
            best = metric
            best_t = cur_t
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        tr_loss = run / max(1, len(ds_tr))
        tag = "ValAUPRC" if binarize else "ValMacroF1"
        print(f"[{model_tag}][{label}] Epoch {ep:02d} | TrainLoss={tr_loss:.4f} | {tag}={metric:.4f} | Best={best:.4f}")

        if PATIENCE and wait >= PATIENCE:
            print(f"[{model_tag}] early stop @ {ep}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test
    model.eval()
    P, Y = [], []
    with torch.no_grad():
        for feats, yb in dl_te:
            feats = [t.to(device) for t in nan_guard(*feats)]
            with autocast("cuda", enabled=torch.cuda.is_available()):
                logits = model(*feats) if len(feats) > 1 else model(feats[0])
                prob = torch.softmax(logits, dim=1)
            P.append(prob.cpu())
            Y.append(yb)
    P = torch.cat(P).numpy() if P else np.zeros((0, K), dtype=float)
    Y = torch.cat(Y).numpy() if Y else np.zeros((0,), dtype=int)

    out = {"model": model_tag, "label": label, "binarize": binarize, "best_thresh": best_t}

    if binarize:
        ypred = (P[:, 1] >= best_t).astype(int) if P.size else np.array([], dtype=int)
        acc = accuracy_score(Y, ypred) if Y.size else 0.0
        f1b = f1_score(Y, ypred) if Y.size else 0.0
        auc = roc_auc_score(Y, P[:, 1]) if Y.size else 0.0
        ap = average_precision_score(Y, P[:, 1]) if Y.size else 0.0
        cm = confusion_matrix(Y, ypred).tolist() if Y.size else [[0, 0], [0, 0]]

        out.update({"acc": acc, "f1": f1b, "auc": auc, "auprc": ap, "cm": cm})

        # Persist curves (for identical-scale overlays)
        if Y.size:
            save_curves(model_tag=model_tag, label=label, y_true=Y, y_score=P[:, 1], outdir=outdir)
            prec, rec, _ = precision_recall_curve(Y, P[:, 1])
            plt.figure()
            plt.plot(rec, prec, label=f"{model_tag} (AP={ap:.3f})")
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"{model_tag}__{label}__PR.png"), dpi=300)
            plt.close()
    else:
        yhat = np.argmax(P, 1) if P.size else np.array([], dtype=int)
        acc = accuracy_score(Y, yhat) if Y.size else 0.0
        bacc = balanced_accuracy_score(Y, yhat) if Y.size else 0.0
        f1m = f1_score(Y, yhat, average="macro") if Y.size else 0.0
        cm = confusion_matrix(Y, yhat).tolist() if Y.size else [[0, 0], [0, 0]]
        out.update({"acc": acc, "bacc": bacc, "f1_macro": f1m, "cm": cm})

    with open(os.path.join(outdir, f"{model_tag}__{label}__metrics.json"), "w") as f:
        json.dump(out, f, indent=2)

    return out


def run_all_labels():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: Dict[str, Dict] = {}
    model_tags = list(MODEL_SPECS.keys())

    for label_name, K in LABEL_DICT.items():
        print(f"\n=== Running all models for {label_name.upper()} (K={K}) ===")
        results[label_name] = {}

        for tag in model_tags:
            r = run_one(label_name, K, tag, device)
            if r is not None:
                results[label_name][tag] = r

        # Overlays for binary labels only
        if K == 2:
            outdir = os.path.join(ROOT_OUT, "overlays", label_name)
            os.makedirs(outdir, exist_ok=True)
            collect_and_plot_overlays(
                curves_root=ROOT_OUT,
                label=label_name,
                model_tags=model_tags,
                outdir=outdir,
            )

    with open(os.path.join(ROOT_OUT, "summary_all_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


# ----------------------------- Entry -----------------------------

if __name__ == "__main__":
    run_all_labels()
