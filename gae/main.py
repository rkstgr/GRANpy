from __future__ import division
from __future__ import print_function

import time
import os

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
import numpy as np
import scipy.sparse as sp

from optimizer import OptimizerAE, OptimizerVAE
from input_data import load_data
from model import GCNModelAE, GCNModelVAE
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges, gen_crossval_edges
from train import train_model, predict_adj, get_scores

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
flags.DEFINE_integer('crossvalidation', 1, 'Whether to use crossvalidation (1) or not (0).')

model_str = FLAGS.model
dataset_str = FLAGS.dataset
model_timestamp = time.strftime("%Y%m%d_%H%M%S") + '_' + dataset_str

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

acc_cv, ap_cv, roc_cv = ([] for i in range(3))

sess = tf.compat.v1.Session()
feed_dict = None
iterations = 1 + FLAGS.crossvalidation * (len(adj) - 1)

for cv_set in range(iterations):
    print("\nCV run " + str(cv_set+1) + " of " + str(iterations) + "...")
    # Construct feed dictionary
    feed_dict = construct_feed_dict(adj_norm[cv_set], adj_label[cv_set], features, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    # Train model
    acc_last, ap_last, roc_last = train_model(adj_orig, FLAGS, [x[cv_set] for x in crossval_edges],
                                        placeholders, opt, sess, model, feed_dict, model_str, model_timestamp)
    for x,l in zip([acc_last, ap_last, roc_last], [acc_cv, ap_cv, roc_cv]):
        l.append(x)

#Save last predicted adj matrix
adj_pred = predict_adj(feed_dict, sess, model, model_timestamp, placeholders, save_adj=True)

#Plot ROC curve
_, _, test_loss, test_acc, test_ap, test_roc = get_scores(adj_pred, adj_orig, test_edges, test_edges_false, model_timestamp, viz_roc=True)
_, _, _, random_acc, random_ap, random_roc = get_scores(np.array(adj[0].todense()), adj_orig, test_edges, test_edges_false, (model_timestamp + "_random"), viz_roc=True, random=True)

if FLAGS.crossvalidation:
    cv_str = str(iterations) + " fold CV "
else:
    cv_str = "Validation "
print("\n" + cv_str + "ROC AUC score: " + str(np.round(np.mean(roc_cv),2)))
print(cv_str + "Average Precision: " + str(np.round(np.mean(ap_cv),2)))
print(cv_str + "Accuracy: " + str(np.round(np.mean(acc_cv),2)))

print('\nTest ROC score: ' + str(np.round(test_roc,2)))
print('Test AP score: ' + str(np.round(test_ap,2)))
print('Test accuracy: ' + str(np.round(test_acc,2)))

print('\nControl ROC score: ' + str(np.round(random_roc,2)))
print('Control AP score: ' + str(np.round(random_ap,2)))
print('Control accuracy: ' + str(np.round(random_acc,2)))

