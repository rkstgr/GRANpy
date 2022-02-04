from __future__ import division
from __future__ import print_function

import json
import random
import time
import os
import warnings

# Train on CPU (hide GPU) due to memory constraints
import wandb

os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
import numpy as np
import scipy.sparse as sp

from optimizer import OptimizerAE, OptimizerVAE
from input_data import load_data
from model import GCNModelAE, GCNModelVAE
from preprocessing import preprocess_graph, sparse_to_tuple, gen_train_val_test_sets
from train import train_test_model
from outputs import save_adj


def set_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


MODEL_SEED = None
DATA_SEED = None

if MODEL_SEED is not None:
    set_seeds(MODEL_SEED)
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.random.set_random_seed(MODEL_SEED)
    random.seed()
    np.random.seed(MODEL_SEED)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('verbose', 1, 'Verbosity of output from low (0) to high (1)')

flags.DEFINE_integer("model_seed", None, "Seed for model weights initialisation")
flags.DEFINE_integer("data_seed", None, "Seed used for data train/val splitting")

flags.DEFINE_float('learning_rate', 0.0002, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 100, 'Number of max epochs to train.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 48, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('early_stopping', 5, 'Tolerance for early stopping (# of epochs).')

flags.DEFINE_string('model', 'gcn_ae', 'Model string.')
flags.DEFINE_float('ratio_val', 0.2, 'Ratio of edges used for validation metrics.')
flags.DEFINE_float('ratio_test', 0.1, 'Ratio of edges used for test metrics.')
flags.DEFINE_integer('balanced_metrics', 1, 'Whether to use balanced metrics (1) or not (0).')

flags.DEFINE_string('dataset', 'gasch_GSE102475', 'Dataset file name.')
flags.DEFINE_string('ground_truth', 'yeast_chipunion_KDUnion_intersect', 'Gold standard edges file name.')
flags.DEFINE_string('inFilePath', None, 'Input Files path')
flags.DEFINE_string('outFilePath', None, 'Output Files path')

flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')
flags.DEFINE_integer('random_prior', 0, 'When prior adjacency matrix should be set to random matrix (1) or not (0).')

flags.DEFINE_integer('crossvalidation', 0, 'Whether to use crossvalidation (1) or not (0).')
flags.DEFINE_integer('hp_optimization', 0, 'Whether to start the hyperparameter optimization run (1) or not (0).')

print("features", FLAGS.features)
wandb.init(project="GranPy", tags=["feature_test"], group="voltaire", config=FLAGS.flag_values_dict())

config = wandb.config

print(config)

model_str = config.model
model_timestamp = time.strftime("%Y%m%d_%H%M%S") + '_' + config.dataset + '_' + config.ground_truth

if config.verbose == 0:
    warnings.filterwarnings("ignore")

if not os.path.exists('logs'):
    os.mkdir('logs')
if not os.path.exists('logs/outputs'):
    os.mkdir('logs/outputs')
if not os.path.exists('logs/training_plots'):
    os.mkdir('logs/training_plots')

# Load data
if config.inFilePath is None:
    norm_expression_path = 'data/normalized_expression/' + config.dataset + '.csv'
    gold_standard_path = 'data/gold_standards/' + config.ground_truth + '.txt'
else:
    norm_expression_path = config.inFilePath + 'ExpressionData' + '.csv'
    gold_standard_path = config.inFilePath + 'PriorNetwork' + '.txt'

adj, features, gene_names = load_data(norm_expression_path, gold_standard_path, model_timestamp, config.random_prior)

# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

rng = np.random.RandomState(seed=DATA_SEED)  # this controls the split
adj_train, crossval_edges, test_edges, test_edges_false = gen_train_val_test_sets(adj_orig,
                                                                                  config.crossvalidation,
                                                                                  config.balanced_metrics,
                                                                                  config.ratio_val,
                                                                                  config.ratio_test,
                                                                                  rng=rng
                                                                                  )

adj = adj_train

if config.features == 0:
    features = sp.identity(features.shape[0])  # featureless

# Some preprocessing
adj_norm = [preprocess_graph(m) for m in adj]

adj_label = [(m + sp.eye(m.shape[0])) for m in adj_train]
# np.savetxt('logs/outputs/' + model_timestamp + '_adj_train.csv', adj_label[-1].toarray(), delimiter=";")
adj_label = [sparse_to_tuple(m) for m in adj_label]

features = sparse_to_tuple(features.tocoo())


def build_tf_graph(model_str, features, adj):
    # Define placeholders
    placeholders = {
        'features': tf.compat.v1.sparse_placeholder(tf.float32),
        'adj': tf.compat.v1.sparse_placeholder(tf.float32),
        'adj_orig': tf.compat.v1.sparse_placeholder(tf.float32),
        'dropout': tf.compat.v1.placeholder_with_default(0., shape=())
    }

    num_features = features[2][1]
    features_nonzero = features[1].shape[0]
    num_nodes = adj[0].shape[0]

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

    return placeholders, model, opt


# Build, train and test model
adj_pred = None

placeholders, model, opt = build_tf_graph(model_str, features, adj)
print(model)
model_timestamp = model_timestamp + "_" + model_str + "_hid1-" + str(config.hidden1) + "_hid2-" + str(
    config.hidden2) + "_lr-" + str(config.learning_rate)
_, _, _, adj_pred = train_test_model(adj_norm, adj_label, features, adj_orig, config, crossval_edges,
                                     placeholders, opt, model, model_str, model_timestamp,
                                     adj, test_edges, test_edges_false)

# Save output adj matrix and gene interaction list
save_adj(adj_pred, config.outFilePath, model_timestamp, gene_names)
