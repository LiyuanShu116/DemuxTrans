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

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import label_binarize

# X_test = pd.read_feather("./data/own/X_test.feather").to_numpy()
# Y_test = pd.read_feather("./data/own/Y_test.feather").to_numpy().squeeze()
X_test = pd.read_feather("./data/quipunet/X_test.feather").to_numpy()
Y_test = pd.read_feather("./data/quipunet/Y_test.feather").to_numpy().squeeze()

X_test = torch.from_numpy(X_test).float()
X_test = X_test.reshape((-1, 1, 700))
test_dataset = BarcodeDataset(X_test, Y_test)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, drop_last=False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_Demuxtrans = Model_TCN(
        num_layers=3,
        d_model=256,
        num_heads=4,
        d_ff=1024,
        num_classes=8,
        #hidden_dim=128,
        dropout=0.1,
        num_channel=48
)
model_Demuxtrans.load_state_dict(torch.load('./model_saved/data1_all.pth'))
model_Demuxtrans.to(device)

# model_Demuxtrans = Model_TCN(
#         num_layers=3,
#         d_model=256,
#         num_heads=8,
#         d_ff=1024,
#         num_classes=4,
#         #hidden_dim=128,
#         dropout=0.1,
#         num_channel=64
# )
# model_Demuxtrans.load_state_dict(torch.load('./model_saved/data3_head_8.pth'))
# model_Demuxtrans.to(device)

def test_with_multiclass_roc(epoch, model, test_loader, device, save_path, best_acc=0.0, num_classes=8):
    model.eval()
    all_probs = []  # To store predicted probabilities for all classes
    all_labels = []  # To store ground truth labels

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device, dtype=torch.float32)
            labels = labels.to(device)
            outputs = model(inputs)

            # Get probabilities using softmax
            probs = torch.softmax(outputs, dim=1)

            all_probs.extend(probs.cpu().numpy())  # Save probabilities
            all_labels.extend(labels.cpu().numpy())  # Save true labels

    # Convert lists to numpy arrays
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Binarize the true labels for multiclass
    all_labels_bin = label_binarize(all_labels, classes=np.arange(num_classes))

    # Compute ROC curve and AUC for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve for each class
    plt.figure(figsize=(8, 6))
    colors = plt.cm.get_cmap('tab10', num_classes)  # Use distinct colors for each class
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], color=colors(i), lw=2, label=f'Barcode {i + 1} (AUC = {roc_auc[i]:.4f})')

    # Plot the diagonal (random guessing) line
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve on Dataset Ⅰ')
    plt.legend(loc='lower right')
    plt.grid(True)
    # plt.show()
    plt.savefig("ROC_1", dpi=600, bbox_inches='tight')

    # Print the AUC values for each class with 4 decimal places
    print("AUC for each class:")
    for i in range(num_classes):
        print(f"Class {i + 1}: {roc_auc[i]:.4f}")

    return roc_auc

roc_auc = test_with_multiclass_roc(epoch=1, model=model_Demuxtrans, test_loader=test_loader, device='cuda', save_path='./best_model.pth', num_classes=8)
print(f'ROC AUC for each class: {roc_auc}')

