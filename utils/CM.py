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
from sklearn.metrics import confusion_matrix

sns.set_theme(style="whitegrid", context="talk")  # 类似 ggplot 的白色网格主题

# X_test = pd.read_feather("../data/own/X_test.feather").to_numpy()
# Y_test = pd.read_feather("../data/own/Y_test.feather").to_numpy().squeeze()

X_test = pd.read_feather("../data/quipunet/X_test.feather").to_numpy()
Y_test = pd.read_feather("../data/quipunet/Y_test.feather").to_numpy().squeeze()

X_test = torch.from_numpy(X_test).float()
X_test = X_test.reshape((-1, 1, 700))
# X_test = signal2image(X_test)

test_dataset = BarcodeDataset(X_test, Y_test)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, drop_last=False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
# model_Demuxtrans.load_state_dict(torch.load('../model_saved/data3_head_8.pth'))
# model_Demuxtrans.to(device)

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
model_Demuxtrans.load_state_dict(torch.load('../model_saved/data1_all.pth'))
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



def test_with_confusion_matrix( model, test_loader, device,  num_classes=8):
    model.eval()
    all_preds = []  # To store all predictions
    all_labels = []  # To store all true labels

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device, dtype=torch.float32)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)  # Get the predicted class labels

            all_preds.extend(predicted.cpu().numpy())  # Save predictions
            all_labels.extend(labels.cpu().numpy())  # Save true labels

    # Convert lists to numpy arrays for calculation
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute and plot the confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=np.arange(num_classes))  # Compute confusion matrix
    plot_confusion_matrix_percentage(cm, num_classes)


def plot_confusion_matrix_percentage(cm, num_classes, title='DemuxTrans'):
    # Convert counts to percentages
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(12, 9))
    sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='GnBu',
                xticklabels=np.arange(1, num_classes + 1),
                yticklabels=np.arange(1, num_classes + 1),
                cbar_kws={'format': '%.0f%%'})
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(title)
    # plt.show()
    plt.savefig("CM_DemuxTrans", dpi=600)

test_with_confusion_matrix(model=model_Demuxtrans, test_loader=test_loader, device=device, num_classes=8)