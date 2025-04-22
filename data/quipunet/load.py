import numpy as np
import pandas as pd

import keras
from Quipu.tools import normaliseLength, noiseLevels

dataset = pd.concat([
    pd.read_hdf("dataset_part1.hdf5"),
    pd.read_hdf("dataset_part2.hdf5")
])
datasetTestEven = pd.read_hdf("datasetTestEven.hdf5")
datasetTestOdd = pd.read_hdf("datasetTestOdd.hdf5")
datasetTestMix = pd.read_hdf("datasetTestMix.hdf5")
datasetWithAntibodies = pd.concat([
    pd.read_hdf("datasetWithAntibodies_part1.hdf5"),
    pd.read_hdf("datasetWithAntibodies_part2.hdf5")
])
datasetExtra = pd.read_hdf("datasetExtra.hdf5")

# Hyperparameters
hp = {
    "traceLength": 700,
    "traceTrim": 0,
    "barcodes": 8,  # distinct barcode count
    "normalise_levels": True,  # wherther to normalise experiments per batch before feetingh into NN
}

# barcode binnary encoding: int(barcode,2)
barcodeEncoding = {
    "000": 0,
    "001": 1,
    "010": 2,
    "011": 3,
    "100": 4,
    "101": 5,
    "110": 6,
    "111": 7
}

# reverse encoding
barcodeEncodingReverse = {v: k for k, v in barcodeEncoding.items()}

def prepareTraces(dataset):
    "trims, clips, and reformats the traces"
    traces = dataset.trace
    traces_uniform = traces.apply(lambda x: normaliseLength(x, length=hp["traceLength"], trim=hp["traceTrim"]))
    if hp["normalise_levels"]:
        traces_normalised = - traces_uniform / dataset.UnfoldedLevel
        return np.vstack(traces_normalised)
    else:
        return np.vstack(traces_uniform)

# barcode->number->onehot
def prepareLabels(dataset):
    "prepare barcode labels for training and testing"
    # for barcodes we use one shot encoding
    return barcodeToOneHot(dataset.barcode)

def barcodeToNumber(barcode):
    "translates the barcode string into number"
    if len(np.shape(barcode)) == 0:
        return barcodeEncoding[barcode]
    elif len(np.shape(barcode)) == 1:
        fn = np.vectorize(lambda key: barcodeEncoding[key])
        return fn(barcode)
    elif len(np.shape(barcode)) == 2 and np.shape(barcode)[1] == 1:
        return barcodeToNumber(np.reshape(barcode, (-1,)))
    else:
        raise ValueError("Error: wrong input recieved: " + str(barcode))


def numberToBarcode(number):
    "number to barcode string"
    if len(np.shape(number)) == 0:
        return barcodeEncodingReverse[number]
    elif len(np.shape(number)) == 1:
        fn = np.vectorize(lambda key: barcodeEncodingReverse[key])
        return fn(number)
    else:
        raise ValueError("Error: wrong input recieved: " + str(number))


def numberToOneHot(number):
    return keras.utils.to_categorical(number, num_classes=hp["barcodes"])


def oneHotToNumber(onehot):
    if np.shape(onehot) == (hp['barcodes'],):
        return np.argmax(onehot)  # 返回最大值索引
    elif len(np.shape(onehot)) == 2 and np.shape(onehot)[1] == hp['barcodes']:
        return np.apply_along_axis(arr=onehot, func1d=np.argmax, axis=1)
    else:
        raise ValueError("Error: wrong input recieved: " + str(onehot))


def barcodeToOneHot(barcode):
    "barcode string to catogory encoding aka One-Hot"
    return numberToOneHot(barcodeToNumber(barcode))


def oneHotToBarcode(onehot):
    "catogory encoding aka One-Hot to barcode string"
    return numberToBarcode(oneHotToNumber(onehot))


def labelToNumber(barcode):
    raise ValueError("Replace labelToNumber")


def numberToLabel(number):
    raise ValueError("Replace numberToLabel")


def toCategories(barcode):
    raise ValueError("Replace toCategories")


def fromCategories(x):
    raise ValueError("Replace fromCategories")

allDatasets = pd.concat([dataset, datasetExtra, datasetWithAntibodies], ignore_index=True)
allDatasets = allDatasets[allDatasets.Filter]  # clear bad points

testSetIndex = [
    # barcode, nanopore
    ('000', 6),
    ('001', 26),
    ('010', 1159),
    ('011', 35),  # unbound
    ('011', 32),  # bound
    ('100', 1933),
    ('101', 30),
    ('110', 12),
    ('111', 14)
]

testSetSelection = allDatasets[["barcode", "nanopore"]] \
    .apply(tuple, axis=1).isin(testSetIndex)  # axis=1遍历行，axis=0遍历列

testSet = allDatasets[testSetSelection]
trainSet = allDatasets[~testSetSelection]

# prepare data
print("Trained noise levels:",
      noiseLevels(train=trainSet.trace.apply(lambda x: x[20:])))

X_train = prepareTraces(trainSet)  # feature
Y_train_barcode = np.vstack(trainSet.barcode.values)  # label1
Y_train_bound = np.vstack(trainSet.Bound.values)  # label2

# divide the data set
ni_train = int(len(X_train) * 0.96)  # training set    #52525
ni_dev = len(X_train) - ni_train  # dev set         #2189

randomIndex = np.arange(len(X_train))
np.random.shuffle(randomIndex)

X_dev = X_train[randomIndex[ni_train:], :]  # (2189, 700)
#X_dev = normalize(X_dev)
Y_dev_barcode = Y_train_barcode[randomIndex[ni_train:], :]  # (2189, 1)
Y_dev_bound = Y_train_bound[randomIndex[ni_train:], :]  # (2189, 1)
#C_dev = extract_feature(X_dev)  # (2189, 13, 7)

X_train = X_train[randomIndex[:ni_train], :]  # (52525, 700)
#X_train = normalize(X_train)
Y_train_barcode = Y_train_barcode[randomIndex[:ni_train], :]  # (52525, 1)
Y_train_bound = Y_train_bound[randomIndex[:ni_train], :]  # (52525, 1)
#C_train = extract_feature(X_train)  # (52525, 13, 7)

X_test = prepareTraces(testSet)  # (3464, 700)
#X_test = normalize(X_test)
Y_test_barcode = np.vstack(testSet.barcode.values)  # (3464, 1)
Y_test_bound = np.vstack(testSet.Bound.values)  # (3464, 1)
#C_test = extract_feature(X_test)  # (3464, 13, 7)

# prepare categories
Y_train = barcodeToOneHot(Y_train_barcode)
Y_dev = barcodeToOneHot(Y_dev_barcode)
Y_test = barcodeToOneHot(Y_test_barcode)

# estimate class weights to reduce overfitting
Y_train_labels = list(map(oneHotToBarcode, Y_train))
Y_dev_labels = list(map(oneHotToBarcode, Y_dev))
Y_test_labels = list(map(oneHotToBarcode, Y_test))
# labels = np.array(['wrong','000', '001', '010', '011', '100', '101', '110', '111'])
labels = np.array(['000', '001', '010', '011', '100', '101', '110', '111'])

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
X_dev = pd.DataFrame(X_dev)
Y_train = pd.DataFrame(Y_train)
Y_test = pd.DataFrame(Y_test)
Y_dev = pd.DataFrame(Y_dev)

X_train.to_feather('X_train.feather')
X_test.to_feather('X_test.feather')
X_dev.to_feather('X_dev.feather')
Y_train.to_feather('Y_train.feather')
Y_dev.to_feather('Y_dev.feather')
Y_test.to_feather('Y_test.feather')
