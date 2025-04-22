import numpy as np
import pandas as pd
from utils.Quipu.tools import normaliseLength
from pathlib import Path
#
# own dataset
data = pd.read_feather('data/3/4.feather')
data_4 = data[['reference_kmer','pA_signal']]
del data

counter = data_4['reference_kmer'].value_counts()

# Folder that stores the Feather files
directory = Path("3")

# Collect all *.feather files in the directory
feather_files = sorted(directory.glob("*.feather"))

# Read every Feather file and merge them into one DataFrame
combined_data = pd.concat(
    (pd.read_feather(file) for file in feather_files),
    ignore_index=True
)

# Retain only the columns required for downstream analysis
data_3 = combined_data[["reference_kmer", "pA_signal"]].copy()

# Optionally free memory
del combined_data

# ----------------------- Parameters & utilities -----------------------
ROWS_PER_KMER = 66033          # rows to keep for every reference‑k‑mer
KMERS = [
    "ACCTG","ACAAG","ACATG","ACATC","ACTGG","ACCAA","ACAAC","ACAAA",
    "ACCTC","ACCAT","ACCCA","ACCAG","ACATT","ACTCA","ACCCC","ACTGC",
    "ACTTC","ACACA","ACAGA","ACCAC","ACAGG","ACTTG","ACTGT","ACCCT",
    "ACTCC","ACCTT","ACTGA","ACACC","ACCTA","ACAGC","ACTTT","ACTAC",
    "ACTTA","ACAGT","ACAAT","ACCGC","ACTAT","ACTAA","ACCCG","ACACT",
    "ACTCT","ACCGG","ACACG","ACATA","ACCGT","ACTAG","ACCGA","ACTCG"
]
assert len(KMERS) == 48, "Expected 48 reference k‑mers."

# ----------------------- Fetch 48 data frames -------------------------
kmer_frames = [
    data_4.loc[data_4['reference_kmer'] == kmer].head(ROWS_PER_KMER)
    for kmer in KMERS
]

# ----------------------- Split into four logical barcodes -------------
# idx % 4 gives the logical barcode class (0‑3) defined by your mapping
barcode_groups = {0: [], 1: [], 2: [], 3: []}
for idx, df in enumerate(kmer_frames):
    barcode_groups[idx % 4].append(df)

# (A) If you still want *lists* of data frames:
barcode_0_list = barcode_groups[0]
barcode_1_list = barcode_groups[1]
barcode_2_list = barcode_groups[2]
barcode_3_list = barcode_groups[3]

# (B) If you prefer one data frame per barcode, concatenate:
barcode_0 = pd.concat(barcode_groups[0], ignore_index=True)
barcode_1 = pd.concat(barcode_groups[1], ignore_index=True)
barcode_2 = pd.concat(barcode_groups[2], ignore_index=True)
barcode_3 = pd.concat(barcode_groups[3], ignore_index=True)


def merge(barcodes):

    merged_data = {
        'reference_kmer': [],
        'pA_signal': []
    }

    for i in range(barcodes[0].shape[0]):
        reference_kmer = ''.join([barcode.iloc[i]["reference_kmer"] for barcode in barcodes])

        pA_signal = np.concatenate([barcode.iloc[i]["pA_signal"] for barcode in barcodes])

        merged_data['reference_kmer'].append(reference_kmer)
        merged_data['pA_signal'].append(pA_signal)

    result_df = pd.DataFrame(merged_data)

    return result_df

barcode_0 = merge(barcode_0)
barcode_1 = merge(barcode_1)
barcode_2 = merge(barcode_2)
barcode_3 = merge(barcode_3)

barcode_0['y'] = 0
barcode_1['y'] = 1
barcode_2['y'] = 2
barcode_3['y'] = 3

barcodes = [barcode_0, barcode_1, barcode_2, barcode_3]
barcodes = pd.concat(barcodes, ignore_index=True)

barcodes.to_feather('own.feather')

data = pd.read_feather("own.feather")

signals = data.iloc[:, 1].to_numpy()

normalized_signals = []
for signal in signals:
    signal_array = signal
    mean_val = np.mean(signal_array)
    std_val = np.std(signal_array)
    if std_val != 0:
        normalized_signal = (signal_array - mean_val) / std_val
    else:
        normalized_signal = signal_array - mean_val
    normalized_signals.append(np.round(normalized_signal, 3))

hp = {
    "traceLength": 700,
    "traceTrim": 0,
    "barcodes": 8,  # distinct barcode count
    "normalise_levels": True,  # wherther to normalise experiments per batch before feetingh into NN
}

def prepareSignal(signal):
    "Trims, clips, and reformats the signals"
    signal_uniform = np.array([normaliseLength(s, length=hp["traceLength"], trim=hp["traceTrim"]) for s in signal])
    return signal_uniform

X = np.array(normalized_signals, dtype=object)
X = prepareSignal(X)
Y = data.iloc[:, 2].to_numpy()

X = pd.DataFrame(X)
Y = pd.DataFrame(Y)
#
# X.to_feather("X.feather")
# Y.to_feather("Y.feather")

dataset_size = X.shape[0]
indices = np.arange(dataset_size)
np.random.shuffle(indices)

train_size = int(0.24 * dataset_size)
val_size = int(0.08 * dataset_size)
test_size = int(0.08 * dataset_size)

train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:train_size + val_size + test_size]

X_train, Y_train = X.iloc[train_indices], Y.iloc[train_indices]
X_val, Y_val = X.iloc[val_indices], Y.iloc[val_indices]
X_test, Y_test = X.iloc[test_indices], Y.iloc[test_indices]

# X_train = pd.DataFrame(X_train)
# Y_train = pd.DataFrame(Y_train)
# X_val = pd.DataFrame(X_val)
# Y_val = pd.DataFrame(Y_val)
# X_test = pd.DataFrame(X_test)
# Y_test = pd.DataFrame(Y_test)

X_train.to_feather("X_train.feather")
Y_train.to_feather("Y_train.feather")
X_val.to_feather("X_val.feather")
Y_val.to_feather("Y_val.feather")
X_test.to_feather("X_test.feather")
Y_test.to_feather("Y_test.feather")

