import numpy as np
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


def load_data(dataset):
    if dataset not in ['cora','citeseer', 'pubmed']:
        #read csv files
        adj_path = 'data/' + dataset + '_input_adj.csv'
        features_path = 'data/' + dataset + '_input_features.csv'
        adj = np.genfromtxt(adj_path, delimiter=';')
        features = np.genfromtxt(features_path, delimiter=';')

        #preprocess matrices (remove isolated nodes, scale features 0-1, make graph undirected, remove self edges)
        adj, features = crop_isolated_nodes(adj, features)
        features = preprocessing.MinMaxScaler().fit_transform(features)
        adj = preprocess_input_adj(adj, sym=True, diag=0)

        adj = sp.csr_matrix(adj)
        features = sp.csr_matrix(features)
        print("shape of adj matrix: " + str(adj.shape))
        print("shape of features matrix: " + str(features.shape))
        
        return adj, features
    
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

    return adj, features

def crop_isolated_nodes(adj_orig, feat_orig):
    adj_cropped = adj_orig
    feat_cropped = feat_orig
    
    node_degrees = np.sum(adj_orig, axis=0) + np.sum(adj_orig, axis=1)
    isolated_nodes = [i for i,d in enumerate(node_degrees) if d==0]
    np.delete(adj_cropped, isolated_nodes, axis=0)
    np.delete(adj_cropped, isolated_nodes, axis=1)
    np.delete(feat_cropped, isolated_nodes, axis=0)
    
    return(adj_cropped, feat_cropped)

def preprocess_input_adj(adj_orig, sym, diag):
    if sym:
        adj_sym = adj_orig + adj_orig.T
        adj_sym[adj_sym > 1] = 1
    else:
        adj_sym = adj_orig
    if diag is not None:    
        np.fill_diagonal(adj_sym, diag)
    
    return adj_sym
