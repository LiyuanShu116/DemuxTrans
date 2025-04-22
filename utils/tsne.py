import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, recall_score, f1_score, cohen_kappa_score

from model.quipunet.quipunet import QuipuModel
from model.model.multi_scale import Model_TCN
from model.deepbinner.deepbinner import DeepBinner
from model.deeplexion.deeplexion import DeeplexionNet

from Test import test, dev
from dataset import BarcodeDataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import weight_norm
from data.deeplexion.Signal2Image import signal2image

from sklearn.manifold import TSNE
import torch
from sklearn.metrics import accuracy_score, recall_score, f1_score, cohen_kappa_score

X_test = pd.read_feather("./data/own/X_test.feather").to_numpy()
Y_test = pd.read_feather("./data/own/Y_test.feather").to_numpy().squeeze()

# X_test = pd.read_feather("./data/quipunet/X_test.feather").to_numpy()
# Y_test = pd.read_feather("./data/quipunet/Y_test.feather").to_numpy().squeeze()

X_test = torch.from_numpy(X_test).float()
X_test = X_test.reshape((-1, 1, 700))
# X_test = signal2image(X_test)

test_dataset = BarcodeDataset(X_test, Y_test)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, drop_last=False)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# model_Demuxtrans = Model_TCN(
#         num_layers=3,   #比对
#         d_model=256,
#         num_heads=4,
#         d_ff=1024,
#         num_classes=8,
#         #hidden_dim=128,
#         dropout=0.1,
#         num_channel=48
# )
# model_Demuxtrans.load_state_dict(torch.load('./model_saved/data1_all.pth'))

model_Demuxtrans = Model_TCN(
        num_layers=3,   #比对
        d_model=256,
        num_heads=8,
        d_ff=1024,
        num_classes=4,
        #hidden_dim=128,
        dropout=0.1,
        num_channel=64
)
model_Demuxtrans.load_state_dict(torch.load('./model_saved/data3_head_8.pth'))

model_Demuxtrans.to(device)

# model_QuipuNet = QuipuModel(num_class=4)
# model_QuipuNet.load_state_dict(torch.load('./model_saved/quipunet_dataset3.pth'))
# model_QuipuNet.to(device)
#
# model_DeepBinner = DeepBinner(class_count=4)
# model_DeepBinner.load_state_dict(torch.load('./model_saved/deepbinner_dataset3.pth'))
# model_DeepBinner.to(device)

# model_DeeplexionNet = DeeplexionNet(num_classes=4)
# model_DeeplexionNet.load_state_dict(torch.load('./model_saved/deeplexion_dataset3.pth'))
# model_DeeplexionNet.to(device)


def visualize_tsne(features: np.ndarray, labels: np.ndarray, num_classes: int, title: str = "Features",
                   perplexity: int = 30, n_iter: int = 1000, learning_rate: int = 200, save_path: str = None):
    """
    Visualize t-SNE for given features and labels.

    Args:
        features (np.ndarray): Feature matrix for t-SNE.
        labels (np.ndarray): Corresponding class labels.
        num_classes (int): Number of classes.
        title (str): Title of the plot.
        perplexity (int): Perplexity parameter for t-SNE.
        n_iter (int): Number of iterations for optimization.
        learning_rate (int): Learning rate for t-SNE optimization.
        save_path (str): Path to save the plot. If None, the plot is displayed but not saved.
    """
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=n_iter, learning_rate=learning_rate)
    features_2d = tsne.fit_transform(features)

    # Create a color palette
    palette = sns.color_palette("hls", num_classes)

    # Plot the t-SNE results
    plt.figure(figsize=(14, 10))
    for class_id in range(num_classes):
        indices = labels == class_id
        plt.scatter(features_2d[indices, 0], features_2d[indices, 1], label=f'Barcode {class_id+1}', alpha=0.7,
                    color=palette[class_id])

    plt.legend(fontsize=12, loc='center left', bbox_to_anchor=(0.01, 0.9), frameon=True, title="Classes")
    plt.title(title, fontsize=16)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"t-SNE plot saved to {save_path}")
    else:
        plt.show()

def visualize_tsne_with_labels(features: np.ndarray, labels: np.ndarray, num_classes: int, title: str = "t-SNE Plot",
                                perplexity: int = 30, n_iter: int = 8000, learning_rate: int = 200, save_path: str = None):
    """
    Visualize t-SNE with labeled clusters and category annotations.

    Args:
        features (np.ndarray): Feature matrix for t-SNE.
        labels (np.ndarray): Corresponding class labels.
        num_classes (int): Number of classes.
        title (str): Title of the plot.
        perplexity (int): Perplexity parameter for t-SNE.
        n_iter (int): Number of iterations for optimization.
        learning_rate (int): Learning rate for t-SNE optimization.
        save_path (str): Path to save the plot. If None, the plot is displayed but not saved.
    """
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=n_iter, learning_rate=learning_rate)
    features_2d = tsne.fit_transform(features)

    # Create a color palette
    palette = sns.color_palette("hls", num_classes)

    # Plot the t-SNE results
    plt.figure(figsize=(14, 10))
    scatter_plots = []
    for class_id in range(num_classes):
        indices = labels == class_id
        scatter = plt.scatter(features_2d[indices, 0], features_2d[indices, 1], label=f'Barcode {class_id+1}', alpha=0.7,
                               color=palette[class_id])
        scatter_plots.append(scatter)

        # Add text label to cluster center
        cluster_center = features_2d[indices].mean(axis=0)
        plt.text(cluster_center[0], cluster_center[1], f'Barcode {class_id+1}', fontsize=10, fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))

    plt.legend(fontsize=12, loc='center left', bbox_to_anchor=(0.01, 0.9), frameon=True, title="Classes")
    plt.title(title, fontsize=16)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"t-SNE plot saved to {save_path}")
    else:
        plt.show()

def visualize_raw_data(test_loader, device: str, num_classes: int, title: str = "Raw Data", save_path: str = None):
    """
    Visualize raw data using t-SNE.

    Args:
        test_loader (DataLoader): DataLoader for test dataset.
        device (str): Device to process data (e.g., 'cuda' or 'cpu').
        num_classes (int): Number of classes.
        title (str): Title of the plot.
        save_path (str): Path to save the plot. If None, the plot is displayed but not saved.
    """
    all_inputs = []  # To store raw input data
    all_labels = []  # To store true labels

    # Iterate over the test loader to collect raw inputs and labels
    for inputs, labels in test_loader:
        all_inputs.extend(inputs.cpu().numpy())  # Save raw inputs
        all_labels.extend(labels.cpu().numpy())  # Save labels

    # Convert to numpy arrays
    all_inputs = np.array(all_inputs)
    all_labels = np.array(all_labels)

    # Flatten raw inputs if they are multi-dimensional (e.g., time-series or images)
    all_inputs_flat = all_inputs.reshape(all_inputs.shape[0], -1)

    # Perform t-SNE visualization
    visualize_tsne(all_inputs_flat, all_labels, num_classes, title, save_path=save_path)


def test_with_tsne(epoch: int, model, test_loader, device: str, num_classes: int = 4, save_path: str = None):
    """
    Test the model and visualize features using t-SNE.

    Args:
        epoch (int): Current epoch.
        model: PyTorch model to test.
        test_loader (DataLoader): DataLoader for test dataset.
        device (str): Device to process data (e.g., 'cuda' or 'cpu').
        num_classes (int): Number of classes.
        save_path (str): Path to save the plot. If None, the plot is displayed but not saved.
    """
    model.eval()
    all_features = []  # To store all extracted features
    all_labels = []  # To store all true labels

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device, dtype=torch.float32)
            labels = labels.to(device)

            # Extract features instead of final predictions
            outputs = model(inputs)  # Modify this line if needed to get intermediate features

            all_features.extend(outputs.cpu().numpy())  # Store extracted features
            all_labels.extend(labels.cpu().numpy())  # Store true labels

    # Convert lists to numpy arrays
    all_features = np.array(all_features)
    all_labels = np.array(all_labels)

    # Perform t-SNE visualization
    visualize_tsne_with_labels(all_features, all_labels, num_classes, title="Features of Dataset Ⅱ", save_path=save_path)


# Example usage
visualize_raw_data(test_loader=test_loader, device=device, num_classes=4, title="Raw Data of Dataset Ⅱ", save_path="tsne_raw_2.png")

test_with_tsne(epoch=1, model=model_Demuxtrans, test_loader=test_loader, device=device, num_classes=4, save_path="tsne_features_2.png")

