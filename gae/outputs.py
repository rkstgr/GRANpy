import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import numpy as np

def viz_train_val_data(hist_scores, model_str, model_timestamp):
    # Plot training & validation metrics
    loss_train, kl_train, acc_train, ap_train, roc_train, loss_val, acc_val, ap_val, roc_val = hist_scores
    figure, axis = plt.subplots(2,2)

    axis[0, 0].plot(loss_train)
    axis[0, 0].plot(loss_val, color='tab:orange')
    axis[0, 0].set_title('Total loss')
    axis[0, 0].set_xlabel('Epoch')

    axis[0, 1].plot(ap_train)
    axis[0, 1].plot(ap_val, color='tab:orange')
    axis[0, 1].set_ylim([0.7, 1.0])
    axis[0, 1].set_title('Average Precision')
    axis[0, 1].set_xlabel('Epoch')

    axis[1, 0].plot([np.subtract(x1, x2) for (x1, x2) in zip(loss_train, kl_train)])
    axis[1, 0].set_title('Recon loss')
    axis[1, 0].set_xlabel('Epoch')
    if model_str == 'gcn_vae':
        axis2 = axis[1, 0].twinx()
        axis2.plot(kl_train, color='tab:orange')
        axis[1, 0].set_title('Recon/KL loss')

    axis[1, 1].plot(roc_train)
    axis[1, 1].plot(roc_val, color='tab:orange')
    axis[1, 1].set_ylim([0.7, 1.0])
    axis[1, 1].set_title('ROC AUC')
    axis[1, 1].set_xlabel('Epoch')

    figure.tight_layout()
    figure.savefig('results/training/' + model_timestamp + '_training_history.png', dpi=300)

def viz_roc_curve(auc, y_pred, y_true, model_timestamp):
    #Plot ROC curve
    fpr, tpr, _ = roc_curve(y_true,  y_pred)
    fig = plt.figure()
    plt.plot(fpr,tpr,label="auc="+str(np.round(auc,2)))
    plt.title("ROC curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc=4)
    
    fig.savefig('results/training/' + model_timestamp + '_ROC_curve.png', dpi=300)

    
