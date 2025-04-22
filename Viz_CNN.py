import os
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap  # if you switch palettes later

from sklearn.metrics import (
    accuracy_score,   # kept in case you extend the script
    recall_score,
    f1_score,
    cohen_kappa_score
)

from model.model.multi_scale import Model_TCN   # DemuxTrans backbone

# ------------------------- model ------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Model_TCN(
    num_layers=3,
    d_model=256,
    num_heads=4,
    d_ff=1024,
    num_classes=8,
    dropout=0.1,
    num_channel=48,
).to(device)

model.load_state_dict(torch.load("./model_saved/data1_all.pth"))
model.eval()

# ------------------------- data -------------------------------------------------
X_test = (
    pd.read_feather("./data/quipunet/X_test.feather")
      .to_numpy()
      .reshape(-1, 1, 700)
)
y_test = pd.read_feather("./data/quipunet/Y_test.feather").to_numpy().squeeze()

X_test = torch.from_numpy(X_test).float().to(device)

# ------------------------- Grad‑CAM helpers -------------------------------------
activations, gradients = {}, {}

def make_fwd_hook(name):
    """Save forward activations for layer *name*."""
    def hook(_, __, output):
        activations[name] = output        # (1, C, T)
    return hook

def make_bwd_hook(name):
    """Save backward gradients for layer *name*."""
    def hook(_, grad_in, grad_out):
        gradients[name] = grad_out[0]     # (1, C, T), grad_out is a tuple
    return hook

# Register hooks on the embedding conv stack
for tag, layer in {
    "embed"       : model.embed,
    "embed_conv1" : model.embed.conv1,
    "embed_conv2" : model.embed.conv2,
    "embed_conv3" : model.embed.conv3,
}.items():
    layer.register_forward_hook(make_fwd_hook(tag))
    layer.register_backward_hook(make_bwd_hook(tag))

# ------------------------- forward + backward pass ------------------------------
x = X_test[0:1].clone().requires_grad_(True)     # single sample
raw_signal = x[0, 0].cpu().numpy()

output = model(x)
target_class = 0
loss = output[0, target_class]

model.zero_grad()
loss.backward()

# ------------------------- plotting --------------------------------------------
def plot_signal_gradcam(act, grad, layer_label, save_file=None):
    """
    Draw raw signal with color‑coded Grad‑CAM intensity.

    Parameters
    ----------
    act : Tensor, shape (1, C, T)
        Forward activations from the chosen layer.
    grad : Tensor, shape (1, C, T)
        Gradients wrt the chosen layer’s output.
    layer_label : str
        Title label for the figure.
    save_file : Path-like or None
        If provided, figure is saved at 600 DPI.
    """
    # --- 1) Grad‑CAM mask ------------------------------------------------------
    weights  = grad.mean(dim=2, keepdim=True)         # (1, C, 1)
    cam      = (weights * act).sum(dim=1)             # (1, T)
    cam      = F.relu(cam)
    cam      = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam_np   = cam[0].cpu().numpy()

    # --- 2) Average across channels to get a single‑channel signal -------------
    signal_np = act.mean(dim=1)[0].cpu().numpy()

    # --- 3) Build a colored line plot -----------------------------------------
    t = np.arange(signal_np.size)
    points   = np.column_stack([t, signal_np]).reshape(-1, 1, 2)
    segs     = np.concatenate([points[:-1], points[1:]], axis=1)
    seg_cols = (cam_np[:-1] + cam_np[1:]) / 2         # segment color

    lc = LineCollection(segs, cmap="coolwarm", norm=plt.Normalize(0, 1))
    lc.set_array(seg_cols)
    lc.set_linewidth(2)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.add_collection(lc)
    ax.set_xlim(0, t[-1])
    margin = 0.1 * np.abs(signal_np).max()
    ax.set_ylim(signal_np.min() - margin, signal_np.max() + margin)

    plt.colorbar(lc, ax=ax, label="Grad‑CAM intensity")
    ax.set_title(f"Grad‑CAM | {layer_label}")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Signal")

    if save_file:
        Path(save_file).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_file, dpi=600, bbox_inches="tight")
    plt.show()

# Draw Grad‑CAM overlays for each convolution in the embed block
plot_signal_gradcam(activations["embed_conv1"], gradients["embed_conv1"],
                    "Embed‑Conv1", save_file="figures/gradcam_conv1.png")

plot_signal_gradcam(activations["embed_conv2"], gradients["embed_conv2"],
                    "Embed‑Conv2", save_file="figures/gradcam_conv2.png")

plot_signal_gradcam(activations["embed_conv3"], gradients["embed_conv3"],
                    "Embed‑Conv3", save_file="figures/gradcam_conv3.png")

# Fusion of all channels at embedding output
plot_signal_gradcam(activations["embed"].transpose(1, 2),
                    gradients["embed"].transpose(1, 2),
                    "Embed‑Fusion", save_file="figures/gradcam_embed.png")

# ------------------------- raw signal reference plot ---------------------------
t = np.arange(raw_signal.size)
plt.figure(figsize=(12, 4))
plt.plot(t, raw_signal, color="black", linewidth=1.5)
plt.title("Raw input signal")
plt.xlabel("Time step")
plt.ylabel("Signal")
Path("figures").mkdir(exist_ok=True)
plt.savefig("figures/raw_signal.png", dpi=600, bbox_inches="tight")
plt.show()
