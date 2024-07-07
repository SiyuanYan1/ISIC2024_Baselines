import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


def compute_isic_metrics(gt, pred, out_dir, test=False):
    """
    :param gt: (batch,) torch tensor with binary labels
    :param pred: (batch, 2) torch tensor with probabilities for both classes
    :param out_dir: string, directory to save the confusion matrix image
    :param test: boolean, whether this is the test set
    :return: various metrics
    """
    gt_np = gt.cpu().detach().numpy()
    pred_np = pred.cpu().detach().numpy()

    # Get the probabilities for the positive class (assuming it's the second column)
    pred_prob = pred_np[:, 1]

    # Convert probabilities to class predictions
    pred_class = (pred_prob > 0.5).astype(int)
    BACC = balanced_accuracy_score(gt_np, pred_class)
    SEN = recall_score(gt_np, pred_class)
    AUC = roc_auc_score(gt_np, pred_prob)
    SPEC = specificity_score(gt_np, pred_class)
    cm = confusion_matrix(gt_np, pred_class)
    print(cm)

    if test:
        # Plot and save confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.savefig(out_dir + 'confusion_matrix_test.jpg', dpi=600, bbox_inches='tight')
        plt.close()

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    return AUC,SEN, SPEC, BACC


def specificity_score(y_true, y_pred):
    """
    Calculate the specificity score.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)