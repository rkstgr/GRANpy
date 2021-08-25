import numpy as np
import scipy.sparse as sp


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(adj_normalized, adj, features, placeholders):
    # construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj_normalized})
    feed_dict.update({placeholders['adj_orig']: adj})
    return feed_dict


def mask_test_edges(adj,num_val=None, num_test=None):
    # Function to build test set with num_test positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]

    #asymetric metrics
    edge_ratio = (adj.shape[0]*(adj.shape[0] - 1)/2 - edges.shape[0]) / edges.shape[0]

    if num_test is None:
        num_test = int(np.floor(edges.shape[0] * 0.1)) #10% of edges for test set
    if num_val is None:
        num_val = int(np.floor(edges.shape[0] * 0.05)) #5% of edges for validation set

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        if (type(a)==list and not a) or not b.any():
            return False
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges)*edge_ratio:
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges)*edge_ratio:
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_j, idx_i], np.array(test_edges_false)):
            continue
        if ismember([idx_i, idx_j], np.array(test_edges_false)):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    train_edges_false = []
    while len(train_edges_false) < len(train_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_j, idx_i], np.array(test_edges_false)):
            continue
        if ismember([idx_i, idx_j], np.array(test_edges_false)):
            continue
        if ismember([idx_j, idx_i], np.array(val_edges_false)):
            continue
        if ismember([idx_i, idx_j], np.array(val_edges_false)):
            continue
        if train_edges_false:
            if ismember([idx_j, idx_i], np.array(train_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(train_edges_false)):
                continue
        train_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(train_edges_false, edges_all)
    assert ~ismember(val_edges_false, np.array(train_edges_false))
    assert ~ismember(test_edges_false, np.array(train_edges_false))
    assert ~ismember(val_edges_false, np.array(test_edges_false))
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, np.array(train_edges_false), val_edges, np.array(val_edges_false), test_edges, np.array(test_edges_false)

def gen_crossval_edges(adj, crossval):
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false = ([] for i in range(5))
    crossval_edges = [train_edges, train_edges_false, val_edges, val_edges_false]
    num_edges = int((adj.todense().sum()-np.diag(adj.todense()).sum())/2)
    num_test = int(np.floor(num_edges * 0.01)) #10% of edges for test set
    num_val = int(np.floor(num_edges * 0.005)) #5% of edges for validation set

    if crossval:
        _, _, _, val_train_edges, val_train_edges_false, test_edges, test_edges_false = mask_test_edges(adj, num_val=num_edges-num_test, num_test=num_test)
        all_edge_idx = list(range(val_train_edges.shape[0]))

        #positive samples
        np.random.shuffle(all_edge_idx)
        for i in range(int((num_edges-num_test)/num_val)):
            val_edges_idx = all_edge_idx[i*num_val:(i+1)*num_val]
            val_edges_cv = val_train_edges[val_edges_idx]
            train_edges_cv = np.delete(val_train_edges, val_edges_idx, axis=0)
            data = np.ones(train_edges_cv.shape[0])
            adj_train_cv = sp.csr_matrix((data, (train_edges_cv[:, 0], train_edges_cv[:, 1])), shape=adj.shape)
            adj_train_cv = adj_train_cv + adj_train_cv.T
            
            adj_train.append(adj_train_cv)
            train_edges.append(train_edges_cv)
            val_edges.append(val_edges_cv)
        
        #negative samples
        np.random.shuffle(all_edge_idx)
        for i in range(int((num_edges-num_test)/num_val)):
            val_edges_false_idx = all_edge_idx[i*num_val:(i+1)*num_val]
            val_edges_false_cv = val_train_edges_false[val_edges_false_idx]
            train_edges_false_cv = np.delete(val_train_edges_false, val_edges_false_idx, axis=0)

            train_edges_false.append(train_edges_false_cv)
            val_edges_false.append(val_edges_false_cv)
            
    else:
        adj_t, train_e, train_e_false, val_e, val_e_false, test_edges, test_edges_false = mask_test_edges(adj, num_val=num_val, num_test=num_test)
        adj_train.append(adj_t)
        for x, l in zip([train_e, train_e_false, val_e, val_e_false], crossval_edges):
            l.append(x)
            
    print("\n++++++++++++++++++++++++++++++++++++++++++++++++++++")   
    print("# edges in original graph: " + str(num_edges))
    print("# pos edges used for training: " + str(train_edges[0].shape[0]))
    print("# pos edges used for validation: " + str(val_edges[0].shape[0]))
    print("# neg edges used for validation: " + str(val_edges_false[0].shape[0]))
    print("# pos edges used for test: " + str(test_edges.shape[0]))
    print("# neg edges used for test: " + str(test_edges_false.shape[0]))
    print("# nr of cv sets: " + str(len(adj_train)))
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++\n") 
    
    return adj_train, crossval_edges, test_edges, test_edges_false
