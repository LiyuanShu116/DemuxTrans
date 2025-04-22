import numpy as np
import pandas as pd
import os
from collections import Counter
from utils.Quipu.tools import normaliseLength


# deeplexion dataset
directory = '1/'

feather_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.feather')]

combined_data_1 = pd.DataFrame()

for feather_file in feather_files:
    data = pd.read_feather(feather_file)
    combined_data_1 = pd.concat([combined_data_1, data], ignore_index=True)


directory = '2/'

feather_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.feather')]

combined_data_2 = pd.DataFrame()

for feather_file in feather_files:
    data = pd.read_feather(feather_file)
    combined_data_2 = pd.concat([combined_data_2, data], ignore_index=True)

data_pre = pd.concat([combined_data_1, combined_data_2], ignore_index=True)

signal = data_pre.iloc[:, 0].to_numpy()

mean_signals = []
for signal in signal:
    num_full_groups = len(signal) // 30
    if num_full_groups > 0:
        mean_signal = np.mean(signal[:num_full_groups * 30].reshape(-1, 30), axis=1)
        mean_signals.append(np.round(mean_signal, 4))


normalized_signals = []
for signal in mean_signals:
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
Y = data_pre.iloc[:, 2].to_numpy() - 1

X = pd.DataFrame(X)
Y = pd.DataFrame(Y)

X.to_feather("X.feather")
Y.to_feather("Y.feather")

dataset_size = X.shape[0]
indices = np.arange(dataset_size)
np.random.shuffle(indices)
train_size = int(0.8 * dataset_size)
val_size = int(0.1 * dataset_size)
test_size = dataset_size - train_size - val_size

train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:train_size + val_size + test_size]

X_train, Y_train = X.iloc[train_indices], Y.iloc[train_indices]      #(25547, 700)
X_val, Y_val = X.iloc[val_indices], Y.iloc[val_indices]              #(3194, 700)
X_test, Y_test = X.iloc[test_indices], Y.iloc[test_indices]

X_train.to_feather("X_train.feather")
Y_train.to_feather("Y_train.feather")
X_val.to_feather("X_val.feather")
Y_val.to_feather("Y_val.feather")
X_test.to_feather("X_test.feather")
Y_test.to_feather("Y_test.feather")









