import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import torch
import matplotlib.pyplot as plt

def evaluation(model, dataloaders, device):
    y_pred_list = []
    y_test = []
    with torch.no_grad():
        model.eval()
        for i, (X_batch, _) in enumerate(dataloaders['val']):
            y_test.extend(_.tolist())
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            _, y_pred_tags = torch.max(y_test_pred, dim = 1)
            y_pred_list.extend(y_pred_tags.cpu().numpy())
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    class2idx = {
        "ants":0,
        "bees":1,
    }
    idx2class = {v: k for k, v in class2idx.items()}
    confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, y_pred_list)).rename(columns=idx2class, index=idx2class)
    sns.heatmap(confusion_matrix_df, annot=True)
    plt.show()

    print(classification_report(y_test, y_pred_list))