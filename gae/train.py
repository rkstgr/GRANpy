from __future__ import division
from __future__ import print_function

import time
import os
import matplotlib.pyplot as plt

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
import numpy as np
import scipy.sparse as sp

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from optimizer import OptimizerAE, OptimizerVAE
from input_data import load_data
from model import GCNModelAE, GCNModelVAE
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges, gen_crossval_edges

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')

flags.DEFINE_string('model', 'gcn_ae', 'Model string.')
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')

model_str = FLAGS.model
dataset_str = FLAGS.dataset
model_timestamp = time.strftime("%Y%m%d_%H%M%S")

# Load data
adj, features = load_data(dataset_str)

# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

adj_train, crossval_edges, test_edges, test_edges_false = gen_crossval_edges(adj_orig)
adj = adj_train


if FLAGS.features == 0:
    features = sp.identity(features.shape[0])  # featureless

# Some preprocessing
adj_norm = [preprocess_graph(m) for m in adj]

# Define placeholders
placeholders = {
    'features': tf.compat.v1.sparse_placeholder(tf.float32),
    'adj': tf.compat.v1.sparse_placeholder(tf.float32),
    'adj_orig': tf.compat.v1.sparse_placeholder(tf.float32),
    'dropout': tf.compat.v1.placeholder_with_default(0., shape=())
}

num_nodes = adj[0].shape[0]

features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
features_nonzero = features[1].shape[0]

# Create model
model = None
if model_str == 'gcn_ae':
    model = GCNModelAE(placeholders, num_features, features_nonzero)
elif model_str == 'gcn_vae':
    model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)


pos_weight = float(adj[0].shape[0] * adj[0].shape[0] - adj[0].sum()) / adj[0].sum()
norm = adj[0].shape[0] * adj[0].shape[0] / float((adj[0].shape[0] * adj[0].shape[0] - adj[0].sum()) * 2)

# Optimizer
with tf.name_scope('optimizer'):
    if model_str == 'gcn_ae':
        opt = OptimizerAE(preds=model.reconstructions,
                          labels=tf.reshape(tf.sparse.to_dense(placeholders['adj_orig'],
                                                                      validate_indices=False), [-1]),
                          pos_weight=pos_weight,
                          norm=norm)
    elif model_str == 'gcn_vae':
        opt = OptimizerVAE(preds=model.reconstructions,
                           labels=tf.reshape(tf.sparse.to_dense(placeholders['adj_orig'],
                                                                       validate_indices=False), [-1]),
                           model=model, num_nodes=num_nodes,
                           pos_weight=pos_weight,
                           norm=norm)

adj_label = [(m + sp.eye(m.shape[0])) for m in adj_train]
adj_label = [sparse_to_tuple(m) for m in adj_label]


def train_model(FLAGS, adj_norm, adj_label, edges, features, placeholders, opt, sess, feed_dict):
    # Initialize session
    sess.run(tf.compat.v1.global_variables_initializer())

    val_ap, val_roc_score, train_ap, train_roc_score, train_loss, train_acc, train_kl = ([] for i in range(7))

    for epoch in range(FLAGS.epochs):

        t = time.time()

        # Run single weight update
        if model_str == 'gcn_vae':
            outs = sess.run([opt.opt_op, opt.cost, opt.accuracy, opt.kl], feed_dict=feed_dict)
        else:
            outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)

        # Compute metrics
        train_edges, train_edges_false, val_edges, val_edges_false = edges

        avg_cost = outs[1]
        train_loss.append(avg_cost)
        avg_accuracy = outs[2]
        train_acc.append(avg_accuracy)
        if model_str == 'gcn_vae':
            avg_kl = outs[3]
            train_kl.append(avg_kl)

        roc_curr, ap_curr = get_roc_score(val_edges, val_edges_false, sess, feed_dict)
        val_roc_score.append(roc_curr)
        val_ap.append(ap_curr)
        
        roc_train, ap_train = get_roc_score(train_edges, train_edges_false, sess, feed_dict)
        train_roc_score.append(roc_train)
        train_ap.append(ap_train)

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
              "train_acc=", "{:.5f}".format(avg_accuracy), "val_roc=", "{:.5f}".format(val_roc_score[-1]),
              "val_ap=", "{:.5f}".format(ap_curr),
              "time=", "{:.5f}".format(time.time() - t))

    print("Optimization Finished!")

    # Plot training & validation metrics
    figure, axis = plt.subplots(2,2)

    axis[0, 0].plot(train_loss)
    axis[0, 0].set_title('Total loss')
    axis[0, 0].set_xlabel('Epoch')

    axis[0, 1].plot(train_ap)
    axis[0, 1].plot(val_ap, color='tab:orange')
    axis[0, 1].set_ylim([0.7, 1.0])
    axis[0, 1].set_title('Average Precision')
    axis[0, 1].set_xlabel('Epoch')

    axis[1, 0].plot([np.subtract(x1, x2) for (x1, x2) in zip(train_loss, train_kl)])
    axis[1, 0].set_title('Recon loss')
    axis[1, 0].set_xlabel('Epoch')
    if model_str == 'gcn_vae':
        axis2 = axis[1, 0].twinx()
        axis2.plot(train_kl, color='tab:orange')
        axis[1, 0].set_title('Recon/KL loss')

    axis[1, 1].plot(train_roc_score)
    axis[1, 1].plot(val_roc_score, color='tab:orange')
    axis[1, 1].set_ylim([0.7, 1.0])
    axis[1, 1].set_title('ROC AUC')
    axis[1, 1].set_xlabel('Epoch')

    figure.tight_layout()
    plt.savefig('results/training/' + model_timestamp + '_training_history.png', dpi=300)

    return val_ap[-1], val_roc_score[-1]

def get_roc_score(edges_pos, edges_neg, sess, feed_dict,emb=None):
    if emb is None:
        feed_dict.update({placeholders['dropout']: 0})
        emb = sess.run(model.z_mean, feed_dict=feed_dict)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
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

    return roc_score, ap_score

val_ap_cv = []
val_roc_score_cv = []
sess = tf.compat.v1.Session()
feed_dict = None

for cv_set in range(len(adj)-17):
    print("\nCV run " + str(cv_set+1) + " of " + str(len(adj)) + "...")
    # Construct feed dictionary
    feed_dict = construct_feed_dict(adj_norm[cv_set], adj_label[cv_set], features, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    # Train model
    val_ap, val_roc_score = train_model(FLAGS, adj_norm[cv_set], adj_label[cv_set], [x[cv_set] for x in crossval_edges], features, placeholders, opt, sess, feed_dict)
    val_ap_cv.append(val_ap)
    val_roc_score_cv.append(val_roc_score)

roc_score, ap_score = get_roc_score(test_edges, test_edges_false, sess, feed_dict)

print("\n10 fold CV Average AP: " + str(np.round(np.mean(val_ap_cv),2)))
print("10 fold CV ROC AUC score: " + str(np.round(np.mean(val_roc_score_cv),2)))

print('Test ROC score: ' + str(np.round(roc_score,2)))
print('Test AP score: ' + str(np.round(ap_score,2)))
