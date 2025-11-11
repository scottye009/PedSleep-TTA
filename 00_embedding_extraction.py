import os, sys, warnings
import numpy as np
import torch
import torch.nn as nn
from random import shuffle
from collections import defaultdict
from torch.utils.data import DataLoader
import PedSleepMAE
from utils.misc import setup_seed
from dataloader import HDF5Dataset

warnings.filterwarnings("ignore")

# config
DATA_DIR = "/path/to/hdf5"
CHECKPOINT_FILE = "/path/to/checkpoint.pt"
SAVE_DIR = "./output_embeddings"
SEARCH_LABEL = "apnea_label"

patch_size = 8
mask_ratio = 15
emb_dim = 64
num_head = 4
num_layer = 3
seed = 42
device = "cuda" if torch.cuda.is_available() else "cpu"
num_patches = int(3840 / patch_size)

setup_seed(seed)

# files
files = [os.path.join(DATA_DIR, x) for x in os.listdir(DATA_DIR) if x.endswith(".hdf5")]
shuffle(files)
if len(files) == 0:
    print("no hdf5 found"); sys.exit(1)

# model
model = PedSleepMAE(
    batch_size=len(files),
    patch_size=patch_size,
    mask_ratio=mask_ratio,
    emb_dim=emb_dim,
    num_head=num_head,
    num_layer=num_layer
).to(device)
ckpt = torch.load(CHECKPOINT_FILE, map_location=device, weights_only=True)
model.load_state_dict(ckpt["state_dict"])

print("device:", device)

# data
batch_size = 100
dataset = HDF5Dataset(files, SEARCH_LABEL)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# run
os.makedirs(SAVE_DIR, exist_ok=True)
pool = nn.AdaptiveMaxPool1d(1)
embeds, labels = defaultdict(list), defaultdict(list)

for b, (signal, label, ids) in enumerate(loader):
    print("batch", b)
    with torch.no_grad():
        signals = signal.squeeze().float().to(device)
        labels_b = label.squeeze().float().to(device)
        n, _, _ = signals.shape
        feats, _ = model.encoder(signals)
        feats = feats[:, :, 1:, :]
        feats = feats.reshape(-1, num_patches, emb_dim)
        pooled = pool(feats).reshape(n, -1)
        for i in range(n):
            sid = ids[i]
            embeds[sid].append(pooled[i].cpu().numpy())
            labels[sid].append(labels_b[i].cpu().numpy().reshape(1))

# save
for sid in embeds.keys():
    e = np.vstack(embeds[sid])
    l = np.concatenate(labels[sid], axis=0)
    np.save(os.path.join(SAVE_DIR, f"{sid}_embeddings.npy"), e)
    np.save(os.path.join(SAVE_DIR, f"{sid}_labels.npy"), l)
    print("saved", sid, e.shape, l.shape)
