import tensorflow as tf
import numpy as np
import time
import scipy.sparse as sp

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from outputs import viz_train_val_data, viz_roc_curve

def train_model(adj_orig, FLAGS, edges, placeholders, opt, sess, model, feed_dict, model_str, model_timestamp):
    # Initialize session
    sess.run(tf.compat.v1.global_variables_initializer())

    loss_train, kl_train, acc_train, ap_train, roc_train, loss_val, acc_val, ap_val, roc_val = ([] for i in range(9))
    hist_scores = [loss_train, kl_train, acc_train, ap_train, roc_train, loss_val, acc_val, ap_val, roc_val]

    #initial metrics
    train_edges, train_edges_false, val_edges, val_edges_false = edges
    adj_pred = predict_adj(feed_dict, sess, model, model_timestamp, placeholders)
    _, _, train_loss, train_acc, train_ap, train_roc = get_scores(adj_pred, adj_orig, train_edges, train_edges_false, model_timestamp)
    _, _, val_loss, val_acc, val_ap, val_roc = get_scores(adj_pred, adj_orig, val_edges, val_edges_false, model_timestamp)
    train_kl = None

    scores = [train_loss, train_kl, train_acc, train_ap, train_roc, val_loss, val_acc, val_ap, val_roc]    
    for x, l in zip(scores, hist_scores):
        l.append(x)

    for epoch in range(FLAGS.epochs):

        t = time.time()

        # Run single weight update
        if model_str == 'gcn_vae':
            outs = sess.run([opt.opt_op, opt.cost, opt.accuracy, opt.kl], feed_dict=feed_dict)
        else:
            outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)

        # Compute metrics
        adj_pred = predict_adj(feed_dict, sess, model, model_timestamp, placeholders)

        ctrl_cost = outs[1]
        ctrl_accuracy = outs[2]

        if model_str == 'gcn_vae':
            train_kl = outs[3]
        else:
            train_kl = 0

        _, total_train_acc, train_loss, train_acc, train_ap, train_roc = get_scores(adj_pred, adj_orig, train_edges, train_edges_false, model_timestamp)
        _, _, val_loss, val_acc, val_ap, val_roc = get_scores(adj_pred, adj_orig, val_edges, val_edges_false, model_timestamp)
        
        scores = [train_loss, train_kl, train_acc, train_ap, train_roc, val_loss, val_acc, val_ap, val_roc]
        for x, l in zip(scores, hist_scores):
            l.append(x)
        
        print("Epoch:", '%04d' % (epoch + 1),
              "train_loss=", "{:.5f}".format(train_loss),
              #"train_loss_control=", "{:.5f}".format(ctrl_cost),
              #"recon loss=", "{:.5f}".format(train_loss-train_kl), "kl_loss=", "{:.5f}".format(train_kl),
              "val_loss=", "{:.5f}".format(val_loss), 
              #"train_acc_control=", "{:.5f}".format(ctrl_accuracy),
              "total_train_acc=", "{:.5f}".format(total_train_acc),
              #"train_acc=", "{:.5f}".format(train_acc),
              #"train_ap=", "{:.5f}".format(train_ap), "train_roc=", "{:.5f}".format(train_roc),
              #"val_acc=", "{:.5f}".format(val_acc),
              "val_ap=", "{:.5f}".format(val_ap), "val_roc=", "{:.5f}".format(val_roc),
              "time=", "{:.5f}".format(time.time() - t))

    print("Optimization Finished!")

    # Plot training & validation metrics
    viz_train_val_data(hist_scores, model_str, model_timestamp)

    return acc_val[-1], ap_val[-1], roc_val[-1], acc_val[0], ap_val[0], roc_val[0] 

def predict_adj(feed_dict, sess, model, model_timestamp, placeholders, emb=None, save_adj=False):
    if emb is None:
        feed_dict.update({placeholders['dropout']: 0})
        emb = sess.run(model.z_mean, feed_dict=feed_dict)

    adj_rec = np.dot(emb, emb.T)
    if save_adj:
        np.savetxt('results/graphs/' + model_timestamp + '_adj_pred'  + '.csv', sigmoid(adj_rec), delimiter=";")

    return adj_rec

def get_scores(adj_rec, adj_orig, edges_pos, edges_neg, model_timestamp, viz_roc=False, random=False):

    if random:
        adj_rec = random_adj(adj_rec, (adj_orig.sum()-adj_rec.sum())/2, mode="random_noraml")
      
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    if viz_roc:
        viz_roc_curve(roc_score, preds_all, labels_all, model_timestamp)

    #Total accuracy and loss
    adj_curr = adj_from_edges(edges_pos, adj_orig.shape)
    adj_curr = adj_curr.reshape(1, -1)
    adj_rec = adj_rec.reshape(1, -1)
    
    pos_weight = float(adj_orig.shape[0] * adj_orig.shape[0] - (edges_pos.shape[0] * 2)) / (edges_pos.shape[0] * 2)
    norm = adj_orig.shape[0] * adj_orig.shape[0] / float((adj_orig.shape[0] * adj_orig.shape[0] - (edges_pos.shape[0] * 2)) * 2)
    cost_total = norm * np.mean(weighted_cross_entropy_with_logits(adj_curr, adj_rec, pos_weight))

    correct_prediction = (sigmoid(adj_rec) > 0.5) == adj_curr
    accuracy_total = np.mean(correct_prediction)

    #Subset accuracy and loss
    test_mask = adj_from_edges(np.hstack([edges_pos, edges_neg]), adj_orig.shape, diag=0)
    test_mask = test_mask.reshape(1, -1)
    accuracy = np.mean(correct_prediction[test_mask==1])
    cost = np.mean(weighted_cross_entropy_with_logits(adj_curr[test_mask==1], adj_rec[test_mask==1], 1))
    
    return cost_total, accuracy_total, cost, accuracy, roc_score, ap_score

def weighted_cross_entropy_with_logits(label, pred, pos_weight):
    return ((1 - label) * pred + (1 + (pos_weight - 1) * label) * (np.log(1 + np.exp(-abs(pred))) + np.maximum(-pred, 0)))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def adj_from_edges(edges, shape, diag=1):
    data = np.ones(edges.shape[0])
    adj = sp.csr_matrix((data, (edges[:, 0], edges[:, 1])), shape=shape)
    adj = adj + adj.T
    if diag==1:
        adj = adj + sp.eye(adj.shape[0])

    return np.array(adj.todense())

def random_adj(adj, edges_to_add, mode="add_edges"):
    if mode=="add_edges":
        edges_added = 0
        adj[adj==0] = -1
        while edges_added < edges_to_add:
            idx_i = np.random.randint(0, adj.shape[0])
            idx_j = np.random.randint(0, adj.shape[0])
            if idx_i == idx_j:
                continue
            elif adj[idx_i, idx_j] == 1:
                continue
            else:
                adj[idx_i, idx_j] = 1
                adj[idx_j, idx_i] = 1
                edges_added += 1
    elif mode=="random_normal":
        adj = np.random.normal(size=adj.shape)
        
    return adj
                
    
