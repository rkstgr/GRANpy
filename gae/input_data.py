import numpy as np
import pandas as pd
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from sklearn import preprocessing


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data(dataset, ground_truth, model_timestamp):
    if dataset not in ['cora','citeseer', 'pubmed']:
        #read csv files
        #adj_path = 'data/' + dataset + '_input_adj.csv'
        #features_path = 'data/' + dataset + '_input_features.csv'
        #adj = np.genfromtxt(adj_path, delimiter=';')
        #features = np.genfromtxt(features_path, delimiter=';')

        #read input data
        norm_expression_path = 'data/normalized_expression/' + dataset + '.csv'
        norm_expression = pd.read_csv(norm_expression_path, sep=',', header=0, index_col=0)
        #print(norm_expression)
        
        gold_standard_path = 'data/gold_standards/' + ground_truth + '.txt'
        gold_standard = pd.read_csv(gold_standard_path, sep='\t', header=None, index_col=False)
        #print(gold_standard)
        
        adj = pd.DataFrame(0, index=norm_expression.index, columns=norm_expression.index)
        for index, row in gold_standard.iterrows():
            if row.iat[0] in norm_expression.index and row.iat[1] in norm_expression.index:
                adj.at[row.iat[0],row.iat[1]] = 1

        adj = np.array(adj.values)
        features = np.array(norm_expression.values)

        #preprocess matrices
        #(remove isolated nodes, scale features 0-1, make graph undirected, remove self edges, shuffle nodes)
        adj = preprocess_input_adj(adj, sym=True, diag=0)
        adj, features = crop_isolated_nodes(adj, features)
        adj, features, order = shuffle_nodes(adj, features)
        features = preprocessing.StandardScaler().fit_transform(features)
        #features= preprocessing.MinMaxScaler().fit_transform(features)
        
        adj = sp.csr_matrix(adj)
        features = sp.csr_matrix(features)
        print("shape of adj matrix: " + str(adj.shape))
        print("shape of features matrix: " + str(features.shape))
        np.savetxt('logs/outputs/' + model_timestamp + '_preprocessed_adj.csv', adj.toarray(), delimiter=";")
        
        return adj, features, order
    
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    np.savetxt('data/' + dataset + '_adj' + '.csv', adj.toarray(), delimiter=";")
    np.savetxt('data/' + dataset + '_features' + '.csv', features.toarray(), delimiter=";")

    return adj, features, None

def crop_isolated_nodes(adj_orig, feat_orig):
    adj_cropped = adj_orig
    feat_cropped = feat_orig
    
    node_degrees = np.sum(adj_orig, axis=0) + np.sum(adj_orig, axis=1)
    isolated_nodes = [i for i,d in enumerate(node_degrees) if d==0]
    adj_cropped = np.delete(adj_cropped, isolated_nodes, axis=0)
    adj_cropped = np.delete(adj_cropped, isolated_nodes, axis=1)
    feat_cropped = np.delete(feat_cropped, isolated_nodes, axis=0)
    
    return (adj_cropped, feat_cropped)

def shuffle_nodes(adj_orig, feat_orig):
    order = np.arange(adj_orig.shape[0])
    np.random.shuffle(order)
    adj_orig = adj_orig[order, :]
    adj_orig = adj_orig[:, order]
    feat_orig = feat_orig[order, :]
        
    return (adj_orig, feat_orig, order)

def preprocess_input_adj(adj_orig, sym, diag):
    if sym:
        adj_sym = adj_orig + adj_orig.T
        adj_sym[adj_sym > 1] = 1
    else:
        adj_sym = adj_orig
    if diag is not None:
        np.fill_diagonal(adj_sym, diag)
    
    return adj_sym
