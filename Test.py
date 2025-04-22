import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pandas as pd
import math
import os

from torch.utils.data import Dataset, DataLoader
from dataset import BarcodeDataset
from torch.utils.tensorboard import SummaryWriter
from dataset import BarcodeDataset
from data.deeplexion.Signal2Image import signal2image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def dev(epoch, model, dev_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dev_loader:
            inputs = inputs.to(device, dtype=torch.float32)
            labels = labels.to(device)
            outputs = model(inputs)
            #outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            # index_tensor = torch.argmax(labels, dim=1)
            # correct += (predicted == index_tensor).sum().item()
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy_dev: {accuracy}%')
    # writer.add_scalar('accuracy_dev', accuracy, epoch)

def test(epoch, model, test_loader, device, save_path, best_acc=0.0):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device, dtype=torch.float32)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    kappa = cohen_kappa_score(all_labels, all_preds)

    print(f'Epoch: {epoch}, Accuracy: {100 * accuracy:.4f}%, Precision: {100 * accuracy:.4f}%, Recall: {recall:.4f}, F1 Score: {f1:.4f}, Kappa: {kappa:.4f}')

    if best_acc < accuracy:
        best_acc = accuracy
        torch.save(model.state_dict(), save_path)
        print(f"New best accuracy: {100 * best_acc:.4f}%, model saved to {save_path}")

    return best_acc

