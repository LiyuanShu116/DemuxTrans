![](C:\Users\uu\Desktop\own paper\ChatGPT Image 2025年4月22日 21_36_15 (1).png)

# **DemuxTrans**

DemuxTrans is a deep learning framework for accurate barcode demultiplexing using raw nanopore sequencing signals. It combines Transformer and TCN architectures to effectively model both local and global dependencies within time-series data, enabling robust classification of barcoded sequences.

![](C:\Users\uu\Desktop\own paper\图\大图1.png)

# **Project Structure**

```
Demuxtrans/
├── data/                  # Contains raw and preprocessed datasets
├── figures/               # Stores generated charts and visualizations.
├── model/                 # Network architecture definitions (DemuxTrans and other comparison Methods)
├── model_saved/           # Used to save trained model weights
├── utils/                 # Utility functions and helper scripts
├── Train.py               # Main training script for model training (DL method)
├── Train_warpdemux.py     # Training script for Warpdemux (ML method)
├── Test.py                # Script to evaluate model performance
├── Transfer.py            # Script implementing model transfer learning
├── Viz_CNN.py             # Script for visualizing internal features of convolutional neural networks
├── dataset.py             # Module for data loading
└── README.md              # Project documentation file
```

# Requirements

Ensure the following dependencies are installed in your environment:

- Python 3.9 or higher
- Pytorch 2.4.1
- NumPy 1.26.4
- Matplotlib 3.9.2

# Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/LiyuanShu116/Demuxtrans.git
   cd Demuxtrans
   ```

2. Prepare the data:

   Place your dataset in the `data/` directory and preprocess it as needed.

3. Train the model:

   ```bash
   python Train.py
   ```

4. Visualization

   ```bash
   python Viz_CNN.py
   ```

# Data Availability

The DemuxTrans project utilizes six distinct datasets (D1–D6) to evaluate performance across various barcoding strategies and sequencing conditions:

## Dataset D1

D1 originates from the QuipuNet project, which provides a multiplexed single-molecule protein sensing dataset. This dataset includes individually controlled measurements for specific barcodes without any rearranged positions, making it well-suited for training supervised learning models. For more details, refer to the [QuipuNet GitHub repository](https://github.com/kmisiunas/QuipuNet).

## **Dataset D2**

D2 comprises fast5-format data obtained from Oxford Nanopore Technologies (ONT) sequencing. These data are basecalled using ONT’s official Guppy software. Four distinct barcode categories are defined, and the corresponding signals are extracted to create the dataset.

## **Dataset D3**

D3 is constructed using data from the DeePlexiCon training dataset to validate the transferability of DemuxTrans. The DeePlexiCon tool is designed for demultiplexing barcoded direct RNA sequencing reads from ONT. For more information, visit the [DeePlexiCon GitHub repository](https://github.com/Psy-Fer/deeplexicon).

## **Datasets D4-D6**

Datasets D4 to D6 are derived from the HycDemux project, which introduces a hybrid unsupervised approach for accurate barcoded sample demultiplexing in nanopore sequencing. These synthetic datasets are generated using DeepSimulator and are subsets of a larger dataset with varying numbers of barcodes.

Please note that only test data is included in this GitHub repository. The fully reprocessed datasets used for training and evaluation is available at the following Zenodo repository:

- D1: https://zenodo.org/records/15266289
- D2: https://zenodo.org/records/15266348
- D3: https://zenodo.org/records/15266170
- D4: https://zenodo.org/records/15266281
- D5: https://zenodo.org/records/15266277
- D6: https://zenodo.org/records/15266267
