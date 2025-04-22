import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    homogeneity_score,
    completeness_score,
    accuracy_score,
)

# ----------------------------------------------------------------------
# 1. Parse *.clstr file → {sample_id: cluster_id}
# ----------------------------------------------------------------------
clstr_file = Path("./results/res_3_95.fa.clstr")

sample2cluster = {}
current_cluster = None

with clstr_file.open() as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue

        if line.startswith(">Cluster"):
            # Example: ">Cluster 0"
            current_cluster = int(line.split()[1])
        else:
            # Example: "0   40nt, >6... *"
            match = re.search(r">(\d+)\.\.\.", line)
            if match:
                sample2cluster[int(match.group(1))] = current_cluster

# ----------------------------------------------------------------------
# 2. Build DataFrame with true / predicted labels
# ----------------------------------------------------------------------
df = pd.DataFrame(
    [(s, c) for s, c in sample2cluster.items()],
    columns=["Sample", "Cluster"],
)

def true_label(sample_id: int, step: int = 1000) -> int:
    """Ground‑truth label via integer division."""
    return (sample_id - 1) // step

df["TrueLabel"] = df["Sample"].apply(true_label)

# ----------------------------------------------------------------------
# 3. External clustering metrics
# ----------------------------------------------------------------------
true_labels = df["TrueLabel"].to_numpy()
pred_labels = df["Cluster"].to_numpy()

homo = homogeneity_score(true_labels, pred_labels)
comp = completeness_score(true_labels, pred_labels)
print(f"Homogeneity : {homo:.4f}")
print(f"Completeness: {comp:.4f}")

# ----------------------------------------------------------------------
# 4. Majority‑vote mapping → sample‑wise accuracy
# ----------------------------------------------------------------------
cluster2label = {
    cid: Counter(sub["TrueLabel"]).most_common(1)[0][0]
    for cid, sub in df.groupby("Cluster")
}

df["PredictedLabel"] = df["Cluster"].map(cluster2label)
sample_acc = accuracy_score(df["TrueLabel"], df["PredictedLabel"])
print(f"Sample‑wise Accuracy (majority vote): {sample_acc:.3f}")
