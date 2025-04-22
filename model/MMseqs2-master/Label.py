from pathlib import Path
import pandas as pd
from collections import Counter
from sklearn.metrics import (
    homogeneity_score,
    completeness_score,
    accuracy_score,
)

# ---------------------------------------------------------------------
# 1. Load clustering output
# ---------------------------------------------------------------------
tsv_file = Path("./results/D3_l/90/clusterRes_cluster.tsv")

df = pd.read_csv(tsv_file, sep="\t", header=None, names=["Cluster", "Sample"])
df = df.astype({"Sample": int, "Cluster": int})

# Ground‑truth label: every 1000 samples form one class
df["TrueLabel"] = (df["Sample"] - 1) // 1000

true_labels = df["TrueLabel"].to_numpy()
pred_labels = df["Cluster"].to_numpy()

# ---------------------------------------------------------------------
# 2. External clustering metrics
# ---------------------------------------------------------------------
homo = homogeneity_score(true_labels, pred_labels)
comp = completeness_score(true_labels, pred_labels)
print(f"Homogeneity : {homo:.4f}")
print(f"Completeness: {comp:.4f}")

# ---------------------------------------------------------------------
# 3. Majority‑vote mapping  (Cluster → PredictedLabel)
# ---------------------------------------------------------------------
cluster2label = (
    df.groupby("Cluster")["TrueLabel"]
      .agg(lambda x: Counter(x).most_common(1)[0][0])
      .to_dict()
)

df["PredictedLabel"] = df["Cluster"].map(cluster2label)

# ---------------------------------------------------------------------
# 4. Sample‑wise accuracy
# ---------------------------------------------------------------------
sample_acc = accuracy_score(df["TrueLabel"], df["PredictedLabel"])
print(f"Sample‑wise Accuracy (majority vote): {sample_acc:.3f}")
