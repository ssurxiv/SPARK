import numpy as np
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def remove_self_loop(adj_):
    #print('remove_self_loop input : ', np.shape(adj))
    """
    remove self loop
    :param adj: (N, V, V)
    :return: (N, V, V)
    """
    adj=adj_.clone()
    num_node = adj.shape[1]
    #print('remove_self_loop num_node : ', num_node)
    if isinstance(adj, torch.Tensor):
        x = torch.arange(0, num_node, device=adj.device)
    else:
        x = np.arange(0, num_node)
    adj = adj.unsqueeze(0) if int(adj.ndim) == 2 else adj
    #print('remove_self_loop adj : ', np.shape(adj))
    adj[:, x, x] = 0
    #print('adj[:, x, x] : ', np.shape(adj))
    return adj
    
def add_self_loop(adj, fill_value: float = 1.):
    """
    add self loop to matrix A
    :param adj: (N,V,V) matrix
    :param fill_value: value of diagonal
    :return: self loop added matrix A
    """
    adj = adj.clone()
    num_node = adj.shape[1]
    if isinstance(adj, torch.Tensor):
        x = torch.arange(0, num_node, device=adj.device)
    else:
        x = np.arange(0, num_node)
    adj[:, x, x] = fill_value
    return adj


def laplacian_norm_adj(A):
    """
    return norm adj matrix
    A` = D'^(-0.5) * A * D'^(-0.5)
    :param A: (N, V, V)
    :return: norm matrix
    """
    A = remove_self_loop(A)
    adj = graph_norm(A)
    # adj = torch.where(torch.isnan(adj), torch.zeros_like(adj), adj)
    return adj

def get_degree_matrix(matrix: np.ndarray, sym=True):  # deprecated
    """
    (deprecated) get degree matrix either tensor or numpy, if you use unsym matrix, out degree and
    :param matrix: (N, V, V)
    :param sym: if sym, True nor False
    :return: D
    """
    matrix[matrix != 0] = 1
    if sym:
        degree = np.sum(matrix, axis=1)
    else:
        out_degree = np.sum(matrix, axis=1)
        in_degree = np.sum(matrix, axis=2)
        degree = out_degree + in_degree
    degree_matrix = np.zeros(matrix.shape)
    ind1, ind2 = np.diag_indices(matrix.shape[1])
    degree_matrix[:, ind1, ind2] = degree
    return degree_matrix

def graph_norm(A):
    D = torch.pow(A.sum(dim=1).clamp(min=1), -0.5) #dim=1
    #print('graph_norm  D: ', np.shape(D))
    # D = torch.diag_embed(D)
    adj = D.unsqueeze(-1) * A * D.unsqueeze(-2)
    #print('graph_norm adj : ', np.shape(adj))
    #print(' D.unsqueeze(-1) : ',  np.shape(D.unsqueeze(-1)))
    #print(' D.unsqueeze(-2) : ',  np.shape(D.unsqueeze(-2)))
    return adj

def index_by_perm(matrix, perm, k=0):
    """
    perm으로 batch indexing을 쉽게 해보자!
    :param matrix: (B, V, F)
    :param perm: (B, V_select)
    :return: permed_matrix
    """
    matrix_temp = torch.cat([torch.index_select(a, k, i).unsqueeze(0) for a, i in zip(matrix, perm)])
    return matrix_temp

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)
        
def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

def __repr__(self):
    # print layer's structure
    return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'

def reset_parameters(self):
    # BN operations
    stdv = 1. / math.sqrt(self.weight.shape[1])
    for weight in self.weight:
        #nn.init.kaiming_uniform_(weight)
        nn.init.xavier_normal_(weight)
    #self.weight.data.uniform_(-stdv, stdv)
    if self.bias is not None:
        nn.init.constant_(self.bias, 0)
        #self.bias.data.uniform_(-stdv, stdv)

        
def chev_conv(self, x, adj_orig):
    # A` * X * W
    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = laplacian_norm_adj(adj_orig)
    adj = add_self_loop(-adj)
    Tx_0 = x[0]
    Tx_1 = x[0]  # Dummy.

    out = torch.matmul(Tx_0, self.weight[0])
    # propagate_type: (x: Tensor, norm: Tensor)
    if self.weight.shape[0] > 1: 
        Tx_1 = torch.matmul(adj, x)
        out = out + torch.matmul(Tx_1, self.weight[1])

    for k in range(2, self.weight.shape[0]):
        Tx_2 = torch.matmul(adj, Tx_1)
        Tx_2 = 2. * Tx_2 - Tx_0
        out = out + torch.matmul(Tx_1, self.weight[k])
        Tx_0, Tx_1 = Tx_1, Tx_2
    if self.bias is not None:
        out += self.bias
    return out