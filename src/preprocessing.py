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

def split_edges(adj, ratio_val, ratio_test):
    
    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_edges = int((adj.todense().sum()-np.diag(adj.todense()).sum())/2)
    num_val = int(np.floor(num_edges * ratio_val))
    num_test = int(np.floor(num_edges * ratio_test))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    return train_edges, val_edges, test_edges


def mask_test_edges(adj,ratio_val, ratio_test, balanced_metrics):
    # Function to build validation/test set with ratio_val/ratio_test ratio of positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    
    def ismember(a, b, tol=5):
        if (type(a)==list and not a) or not b.any():
            return False
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)
    
    train_edges, val_edges, test_edges = split_edges(adj, ratio_val, ratio_test)
    edges_all = sparse_to_tuple(adj)[0]

    if not balanced_metrics:
        neg_adj = sp.csr_matrix(np.ones(adj.shape)-adj)
        train_edges_false, val_edges_false, test_edges_false = split_edges(neg_adj, ratio_val, ratio_test)      

    else:
        test_edges_false = []
        while len(test_edges_false) < len(test_edges):
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
        while len(val_edges_false) < len(val_edges):
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

        #convert lists to numpy arrays
        train_edges_false = np.array(train_edges_false)
        val_edges_false = np.array(val_edges_false)
        test_edges_false = np.array(test_edges_false)

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(train_edges_false, edges_all)
    assert ~ismember(val_edges_false, np.array(train_edges_false))
    assert ~ismember(test_edges_false, np.array(train_edges_false))
    assert ~ismember(val_edges_false, np.array(test_edges_false))
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    ## Check split of edges:
    #np.savetxt('logs/outputs/' + 'train_edges.csv', construct_adj(train_edges, adj.shape).toarray(), delimiter=";")
    #np.savetxt('logs/outputs/' + 'train_edges_false.csv', construct_adj(train_edges_false, adj.shape).toarray(), delimiter=";")
    #np.savetxt('logs/outputs/' + 'val_edges.csv', construct_adj(val_edges, adj.shape).toarray(), delimiter=";")
    #np.savetxt('logs/outputs/' + 'val_edges_false.csv', construct_adj(val_edges_false, adj.shape).toarray(), delimiter=";")
    #np.savetxt('logs/outputs/' + 'test_edges.csv', construct_adj(test_edges, adj.shape).toarray(), delimiter=";")
    #np.savetxt('logs/outputs/' + 'test_edges_false.csv', construct_adj(test_edges_false, adj.shape).toarray(), delimiter=";")

    # NOTE: these edge lists only contain single direction of edge!
    return train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false

def construct_adj(edges, shape):
    # Re-build adj matrix
    data = np.ones(edges.shape[0])
    adj = sp.csr_matrix((data, (edges[:, 0], edges[:, 1])), shape=shape)
    adj = adj + adj.T

    return adj

def gen_train_val_test_sets(adj, crossval, balanced_metrics, ratio_val, ratio_test):
    adj_train, train_edges, train_edges_false, val_edges, val_edges_false = ([] for i in range(5))
    crossval_edges = [train_edges, train_edges_false, val_edges, val_edges_false]
    splits = int(np.round((1-ratio_test)/ratio_val, 0))
    
    if crossval and not balanced_metrics:
        for i in range(5):
            train_e, train_e_false, val_e, val_e_false, test_edges, test_edges_false = mask_test_edges(adj, ratio_val, ratio_test, balanced_metrics)
            adj_train.append(construct_adj(train_e, adj.shape))
            for x, l in zip([train_e, train_e_false, val_e, val_e_false], crossval_edges):
                l.append(x)
            print("generated training/validation set " + str(i) + " of " + str(30) + "...")
            
    elif crossval and balanced_metrics:
        val_train_edges, val_train_edges_false, _ , _, test_edges, test_edges_false = mask_test_edges(adj, 0, ratio_test, balanced_metrics)

        #positive samples
        all_edge_idx = list(range(val_train_edges.shape[0]))
        num_val = np.floor(val_train_edges.shape[0]/splits)
        np.random.shuffle(all_edge_idx)
        for i in range(splits):
            val_edges_idx = all_edge_idx[i*num_val:(i+1)*num_val]
            val_edges_cv = val_train_edges[val_edges_idx]
            train_edges_cv = np.delete(val_train_edges, val_edges_idx, axis=0)
            
            adj_train.append(construct_adj(train_edges_cv, adj.shape))
            train_edges.append(train_edges_cv)
            val_edges.append(val_edges_cv)
        
        #negative samples
        all_false_edge_idx = list(range(val_train_edges_false.shape[0]))
        num_val_false = np.floor(val_train_edges_false.shape[0]/splits)
        np.random.shuffle(all_false_edge_idx)
        for i in range(splits):
            val_edges_false_idx = all_edge_idx[i*num_val_false:(i+1)*num_val_false]
            val_edges_false_cv = val_train_edges_false[val_edges_false_idx]
            train_edges_false_cv = np.delete(val_train_edges_false, val_edges_false_idx, axis=0)

            train_edges_false.append(train_edges_false_cv)
            val_edges_false.append(val_edges_false_cv)
            
    else:
        train_e, train_e_false, val_e, val_e_false, test_edges, test_edges_false = mask_test_edges(adj, ratio_val, ratio_test, balanced_metrics)
        adj_train.append(construct_adj(train_e, adj.shape))
        for x, l in zip([train_e, train_e_false, val_e, val_e_false], crossval_edges):
            l.append(x)
            
    print("\n++++++++++++++++++++++++++++++++++++++++++++++++++++")   
    print("# edges in original graph: " + str(int((adj.todense().sum()-np.diag(adj.todense()).sum())/2)))
    print("# imbalance ratio (non-edges/edges): " + str(int((adj.shape[0]*(adj.shape[0] - 1) - adj.todense().sum() + np.diag(adj.todense()).sum()) / (adj.todense().sum()-np.diag(adj.todense()).sum()))))
    print("# pos edges used for training: " + str(train_edges[0].shape[0]))
    print("# pos edges used for validation: " + str(val_edges[0].shape[0]))
    print("# neg edges used for validation: " + str(val_edges_false[0].shape[0]))
    print("# pos edges used for test: " + str(test_edges.shape[0]))
    print("# neg edges used for test: " + str(test_edges_false.shape[0]))
    print("# nr of cv sets: " + str(len(adj_train)))
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++\n") 
    
    return adj_train, crossval_edges, test_edges, test_edges_false
