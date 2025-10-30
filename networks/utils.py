import torch


def compute_euclid_distance(e):
    e_square = (e ** 2).sum(dim=-1, keepdim=True)  # [B, N, 1]
    dist_squared = e_square + e_square.transpose(1, 2) - 2 * torch.matmul(e, e.transpose(1, 2))  # [B, N, N]
    dist_squared = torch.clamp(dist_squared, min=1e-12)
    return torch.sqrt(dist_squared)

def add_full_rrwp(data, adj, walk_length):
    
    pes = []
    for ids in range(data.shape[0]):
        dt = data[ids].squeeze()
        at = adj[ids].squeeze()
        pe = add_every_rrwp(dt,at, walk_length)
        pes.append(pe)
    return torch.stack(pes)

def add_every_rrwp(data,
                  a,
                  walk_length=8,
                  # attr_name_abs="rrwp",  # name: 'rrwp'
                  # attr_name_rel="rrwp",  # name: ('rrwp_idx', 'rrwp_val')
                  add_identity=True,
                  spd=False,
                  **kwargs
                  ):
    # print(a)
    # adj = data * a
    edge_index = torch.column_stack(torch.where(a)).T.contiguous()
    device = data.device
    num_nodes = data.shape[0]
    edge_weight = torch.abs(data[edge_index[0], edge_index[1]])
    adj = torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes, num_nodes))
    adj = adj.to_dense()
    # Compute D^{-1} A:
    deg_inv = 1.0 / a.sum(dim=1)
    deg_inv[deg_inv == float('inf')] = 0
    adj = adj * deg_inv.view(-1, 1)
    adj = adj.to_dense()

    pe_list = []
    i = 0
    if add_identity:
        pe_list.append(torch.eye(num_nodes, dtype=torch.float, device=device))
        i = i + 1

    out = adj
    pe_list.append(adj)

    if walk_length > 2:
        for j in range(i + 1, walk_length):
            out = out @ adj
            pe_list.append(out)

    pe = torch.stack(pe_list, dim=-1) # n x n x k

    abs_pe = pe.diagonal().transpose(0, 1)  # n x k

    return abs_pe