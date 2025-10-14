import tensorflow as tf
import scipy.sparse as sp
import numpy as np


def normalized_adj(adj):
    """Normalize adjacency matrix"""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    normalized_adj = normalized_adj.astype(np.float32)
    return normalized_adj


def sparse_to_tuple(mx):
    """Convert sparse matrix to tuple format"""
    mx = mx.tocoo()
    coords = np.vstack((mx.row, mx.col)).transpose()
    L = tf.SparseTensor(coords, mx.data, mx.shape)
    return tf.sparse_reorder(L)


def calculate_laplacian(adj, lambda_max=1):
    """Calculate Laplacian matrix - keep original logic unchanged"""
    adj = normalized_adj(adj + sp.eye(adj.shape[0]))
    adj = sp.csr_matrix(adj)
    adj = adj.astype(np.float32)
    return sparse_to_tuple(adj)


def weight_variable_glorot(input_dim, output_dim, name=""):
    """Glorot initialization"""
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform([input_dim, output_dim], minval=-init_range,
                                maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def cal_laplacian(graph):
    """Calculate Laplacian matrix"""
    graph = graph.A
    I = np.eye(graph.shape[0], dtype=graph.dtype)
    graph = graph + I
    D = np.diag(graph.sum(axis=1) ** (-0.5))
    L = I - np.matmul(np.matmul(D, graph), D)
    return L


def cheb_polynomial(laplacian, K):
    """
    Calculate Chebyshev polynomials.

    Args:
        laplacian: Graph Laplacian matrix with shape [v, v].
        K: Number of polynomial orders.

    Returns:
        Multi-order Chebyshev Laplacian matrices with shape [K, v, v].
    """
    N = laplacian.shape[0]
    multi_order_laplacian = np.zeros([K, N, N], dtype=np.float32)
    multi_order_laplacian[0] = np.eye(N, dtype=np.float32)

    if K == 1:
        return multi_order_laplacian
    else:
        multi_order_laplacian[1] = laplacian
        if K == 2:
            return multi_order_laplacian
        else:
            for k in range(2, K):
                multi_order_laplacian[k] = 2 * np.matmul(laplacian, multi_order_laplacian[k - 1]) - \
                                           multi_order_laplacian[k - 2]
    return multi_order_laplacian