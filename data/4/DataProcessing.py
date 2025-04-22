import numpy as np
import glob
import os
from utils.Quipu.tools import normaliseLength
import pandas as pd


# Directory containing raw signal files
FOLDER_PATH = "./D1/RealAmpBarSigsONT12"

# Locate all files that match timeSeries_*.txt
FILE_PATTERN = os.path.join(FOLDER_PATH, "timeSeries_*.txt")
file_list = glob.glob(FILE_PATTERN)
if not file_list:
    raise FileNotFoundError(f"No files found matching pattern: {FILE_PATTERN}")

EXPECTED_LENGTH = 310          # Raw length each signal should have
TRACE_LENGTH = 700             # Length after padding/truncation
BARCODE_DIVISOR = 1000         # Determines barcode label from file index

rows = []
for file in file_list:
    # Derive numeric index and barcode label from filename
    idx = int(os.path.basename(file).split("_")[1].split(".")[0])
    label = idx // BARCODE_DIVISOR

    try:
        signal = np.loadtxt(file)
        if np.isscalar(signal):
            signal = np.array([signal])

        # Skip files with unexpected length
        if signal.shape[0] != EXPECTED_LENGTH:
            continue
    except Exception as e:
        print(f"Error reading {file}: {e}")
        continue

    # Z‑score normalisation (rounded for compact storage)
    mean, std = signal.mean(), signal.std()
    norm_signal = (signal - mean) / std if std != 0 else signal - mean
    norm_signal = np.round(norm_signal, 3)

    rows.append(np.hstack((norm_signal, [label])))

if not rows:
    raise ValueError("No valid data was loaded from the files.")

data = np.vstack(rows)
X_raw, y = data[:, :-1], data[:, -1]

def pad_or_trim(signals, length=TRACE_LENGTH, trim=0):
    """Pad or trim each signal to a uniform length."""
    return np.array([normaliseLength(s, length=length, trim=trim) for s in signals])

X = pad_or_trim(X_raw)

# Save as Feather for fast I/O in downstream steps
pd.DataFrame(X).to_feather("X.feather")
pd.DataFrame(y).to_feather("y.feather")

dataset_size = X.shape[0]
indices = np.arange(dataset_size)
np.random.shuffle(indices)

train_size = int(0.8 * dataset_size)

train_indices = indices[:train_size]
test_indices = indices[train_size:]

X_train, Y_train = X.iloc[train_indices], y.iloc[train_indices]
X_test, Y_test = X.iloc[test_indices], y.iloc[test_indices]

X_train.to_feather("X_train.feather")
Y_train.to_feather("Y_train.feather")
X_test.to_feather("X_test.feather")
Y_test.to_feather("Y_test.feather")




