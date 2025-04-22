import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.metrics import accuracy_score

# ----------------------------------------------------------------------
# 1. Read the clustering‑result text file
# ----------------------------------------------------------------------
data_file = Path("./Results/res_3_0.95.txt")

rows = [
    line.strip().split("\t")
    for line in data_file.read_text().splitlines()
    if line.strip()
]

# ----------------------------------------------------------------------
# 2. Build a DataFrame with sample, cluster, and ground‑truth label
# ----------------------------------------------------------------------
records = []
for cluster_id, sample_entry in rows:
    sample_id = int(sample_entry.lstrip(">"))
    true_label = (sample_id - 1) // 1000          # ground‑truth from sample index
    records.append((sample_id, int(cluster_id), true_label))

df = pd.DataFrame(records, columns=["Sample", "Cluster", "TrueLabel"])

# ----------------------------------------------------------------------
# 3. Majority‑vote mapping: Cluster → PredictedLabel
# ----------------------------------------------------------------------
cluster2label = {
    cid: Counter(sub["TrueLabel"]).most_common(1)[0][0]
    for cid, sub in df.groupby("Cluster")
}

df["PredictedLabel"] = df["Cluster"].map(cluster2label)

# ----------------------------------------------------------------------
# 4. Sample‑wise accuracy
# ----------------------------------------------------------------------
accuracy = accuracy_score(df["TrueLabel"], df["PredictedLabel"])
print(f"Sample‑wise Accuracy (majority vote): {accuracy:.3f}")
