from __future__ import division
from __future__ import print_function

import time
import os

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import numpy as np
import scipy.sparse as sp

from optimizer import OptimizerAE, OptimizerVAE
from input_data import load_data
from model import GCNModelAE, GCNModelVAE
from preprocessing import preprocess_graph, sparse_to_tuple, gen_crossval_edges
from train import train_test_model

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.00005, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 12, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 10, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('early_stopping', 5, 'Tolerance for early stopping (# of epochs).')

flags.DEFINE_string('model', 'gcn_ae', 'Model string.')
flags.DEFINE_string('dataset', 'gasch_GSE102475', 'Dataset file name.')
flags.DEFINE_string('ground_truth', 'yeast_chipunion_KDUnion_intersect', 'Gold standard edges file name.')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')
flags.DEFINE_integer('crossvalidation', 0, 'Whether to use crossvalidation (1) or not (0).')

flags.DEFINE_integer('hp_optimization', 0, 'Whether to start the hyperparameter optimization run (1) or not (0).')

model_str = FLAGS.model
dataset_str = FLAGS.dataset
model_timestamp = time.strftime("%Y%m%d_%H%M%S") + '_' + dataset_str + '_' + FLAGS.ground_truth

# Load data
adj, features, gene_names = load_data(dataset_str, FLAGS.ground_truth, model_timestamp)

# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

adj_train, crossval_edges, test_edges, test_edges_false = gen_crossval_edges(adj_orig, FLAGS.crossvalidation)
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
np.savetxt('logs/outputs/' + model_timestamp + '_adj_train.csv', adj_label[-1].toarray(), delimiter=";")
adj_label = [sparse_to_tuple(m) for m in adj_label]

#Train and test model
if FLAGS.hp_optimization:
    #Hyperparameter Optimization
    HP_NUM_UNITS1 = hp.HParam('num_units1', hp.Discrete([2, 5, 8, 12, 16, 32, 64, 128]))
    HP_RATIO_UNITS2 = hp.HParam('ratio_units2', hp.Discrete([0.1, 0.25, 0.4, 0.65, 0.8]))
    HP_LR = hp.HParam('lr', hp.Discrete([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]))

    session_num = 0
    val_acc, val_ap, val_roc = (tf.Variable(0, dtype=tf.float32) for i in range(3))
    val_acc_sum = tf.summary.scalar('Validation Accuracy', val_acc)
    val_ap_sum = tf.summary.scalar('Validation Average Precision', val_ap)
    val_roc_sum = tf.summary.scalar('Validation ROC AUC', val_roc)
    tb_sess = tf.Session()

    for num_units1 in HP_NUM_UNITS1.domain.values:
      for ratio_units2 in HP_RATIO_UNITS2.domain.values:
        for lr in HP_LR.domain.values:
            hparams = {
                HP_NUM_UNITS1: num_units1,
                HP_RATIO_UNITS2: ratio_units2,
                HP_LR: lr,
            }
            FLAGS.learning_rate = hparams[HP_LR]
            FLAGS.hidden1 = hparams[HP_NUM_UNITS1]
            FLAGS.hidden2 = np.ceil(hparams[HP_RATIO_UNITS2]*hparams[HP_NUM_UNITS1])
            run_name = "run" + str(session_num) + "_" + model_str + "_hid1-" + str(FLAGS.hidden1) + "_hid2-" + str(FLAGS.hidden2) + "_lr-" + str(FLAGS.learning_rate)
            print('--- Starting trial %d' % session_num)
            print({h.name: hparams[h] for h in hparams})
            writer = tf.compat.v1.summary.FileWriter('logs/hparam_tuning/' + model_timestamp +'/' + run_name)
            acc, ap, roc = train_test_model(adj_norm, adj_label, features, adj_orig, FLAGS, crossval_edges,
                                            placeholders, opt, model, model_str, (model_timestamp + '_' + run_name),
                                            adj, test_edges, test_edges_false)
            tb_sess.run(val_acc.assign(acc))
            tb_sess.run(val_ap.assign(ap))
            tb_sess.run(val_roc.assign(roc))
            writer.add_summary(tb_sess.run(val_acc_sum), 1)
            writer.add_summary(tb_sess.run(val_ap_sum), 1)
            writer.add_summary(tb_sess.run(val_roc_sum), 1)
            
            writer.flush()
            writer.close()
            session_num += 1
      
else:
    #Run model with given hyperparameters
    model_timestamp = model_timestamp + "_" + model_str + "_hid1-" + str(FLAGS.hidden1) + "_hid2-" + str(FLAGS.hidden2) + "_lr-" + str(FLAGS.learning_rate)
    _, _, _ = train_test_model(adj_norm, adj_label, features, adj_orig, FLAGS, crossval_edges,
                               placeholders, opt, model, model_str, model_timestamp,
                               adj, test_edges, test_edges_false)
    
