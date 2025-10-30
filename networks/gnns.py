import torch
import torch.nn as nn
import torch.nn.functiona as F

import math
from network.gcn_utils import laplacian_norm_adj, add_self_loop, graph_norm

class Cheb2(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, num_node):
        super().__init__()
        self.gconv1 = myChebConv(in_channels=input_dim, out_channels=hidden_dim,K=2)
        self.gconv2 = myChebConv(in_channels=hidden_dim, out_channels=out_dim,K=2)
        self.lin1 = nn.Linear(in_features=num_node*out_dim, out_features=num_node*64)
        self.lin2 = nn.Linear(in_features=num_node*64 , out_features=num_node*32)
        self.lin3 = nn.Linear(in_features=num_node*32, out_features=2) # 2
        self.bn1 = nn.BatchNorm1d(num_features=num_node*64)
        self.bn2 = nn.BatchNorm1d(num_features=num_node*32)
        self.act = nn.Mish()
        
    def forward(self, x, adj):
        b,_,_ = x.shape
        x = self.act(self.gconv1(x, adj))
        x = self.act(self.gconv2(x, adj))
        # x = self.act(self.gconv(x, adj))
        x = x.reshape((b, -1))
        x = self.act(self.bn1(self.lin1(x)))
        x = self.act(self.bn2(self.lin2(x)))
        out = self.lin3(x)
        return out

class Cheb1(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # self.ln = nn.Linear(input_dim, input_dim)
        self.gconv1 = myChebConv(in_channels=input_dim, out_channels=hidden_dim,K=2)

        self.act = nn.Mish()
        
    def forward(self, x, adj):
        b,_,_ = x.shape
        # x = self.ln(x)
        x = self.act(self.gconv1(x, adj))
        return x

class myChebConv(nn.Module):
    """
    simple GCN layer
    Z = f(X, A) = softmax(A` * ReLU(A` * X * W0)* W1)
    A` = D'^(-0.5) * A * D'^(-0.5)
    """
    def __init__(self, in_channels, out_channels, K=2, bias=True):
        # input
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(K, in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    def reset_parameters(self):
        # BN operations
        stdv = 1. / math.sqrt(self.weight.shape[1])
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
            
    def forward(self, x, adj_orig):
        # A` * X * W
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = laplacian_norm_adj(adj_orig)
        adj = add_self_loop(-adj)
        Tx_0 = x  # T0(x) = 1, 여기서는 k=1시작
        Tx_1 = x  # Dummy.
        out = torch.matmul(Tx_0, self.weight[0])
        # propagate_type: (x: Tensor, norm: Tensor)
        if self.weight.shape[0] > 1:
            Tx_1 = torch.matmul(adj, x)
            out = out + torch.matmul(Tx_1, self.weight[1])
        for k in range(2, self.weight.shape[0]):
            Tx_2 = torch.matmul(adj, Tx_1)
            Tx_2 = 2. * Tx_2 - Tx_0  # Chebyshef polynomial
            out = out + torch.matmul(Tx_2, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2
        if self.bias is not None:
            out += self.bias
        return out

    def __repr__(self):
        # print layer's structure
        return self.__class__.__name__ + ' (' \
               + str(self.in_channels) + ' -> ' \
               + str(self.out_channels) + ')'



class HyperbolicGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias, use_att, local_agg):
        super(HyperbolicGraphConvolution, self).__init__()
        self.c = c_in
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        self.agg = HypAgg(manifold, c_in, out_features, dropout, use_att, local_agg)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)

    def forward(self, x, adj):

        h = self.linear.forward(x)
        h = self.agg.forward(h, adj)
        h = self.hyp_act.forward(h)
    
        return h


class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """
    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        torch.nn.init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        res = self.manifold.mobius_matvec(drop_weight, x, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            # hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            # res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )


class HypAgg(nn.Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold, c, in_features, dropout, use_att, local_agg):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = c

        self.in_features = in_features
        self.dropout = dropout
        self.local_agg = local_agg
        self.use_att = use_att
        if self.use_att:
            self.att = DenseAtt(in_features, dropout)

    def forward(self, x, adj):
        batch_size, node_num, feature_dim = x.size()

        x_tangent = self.manifold.logmap0(x, c=self.c)
        if self.use_att:
            if self.local_agg:
                x_local_tangent = []
                for i in range(node_num):
                    tangent_i = self.manifold.logmap(
                        x[:, i, :].unsqueeze(1).expand(-1, node_num, -1),  # (batch_size, node_num, feature_dim)
                        x, 
                        c=self.c
                    )
                    x_local_tangent.append(tangent_i)
                x_local_tangent = torch.stack(x_local_tangent, dim=1)
                adj_att = self.att(x_tangent, adj)
                att_rep = adj_att.unsqueeze(-1) * x_local_tangent
                support_t = torch.sum(att_rep, dim=2)
                output = self.manifold.expmap(x, support_t, c=self.c)
                # output = self.manifold.proj(self.manifold.expmap(x, support_t, c=self.c), c=self.c)
                return output
            else:
                adj_att = self.att(x_tangent, adj)
                support_t = torch.matmul(adj_att, x_tangent)
        else:
            # 여기 변경
            # adj = laplacian_norm_adj(adj)
            # adj = add_self_loop(adj)
            adj = graph_norm(adj)
            support_t = torch.bmm(adj, x_tangent)
            support_t = self.manifold.proj_tan0(support_t, c=self.c) 
        output = self.manifold.expmap0(support_t, c=self.c)
        # output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)


class HypAct(nn.Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        return self.manifold.expmap0(xt, c=self.c_out)
        # return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )
    
class DenseAtt(nn.Module):
    def __init__(self, in_features, dropout):
        super(DenseAtt, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(2 * in_features, 1, bias=True)
        self.in_features = in_features

    def forward (self, x, adj):
        batch_size, n, d = x.size()
        # x_left : (batch_size, n, n, d)
        x_left = x.unsqueeze(2).expand(-1, -1, n, -1)

        # x_right : (batch_size, n, n, d)
        x_right = x.unsqueeze(1).expand(-1, n, -1, -1)

        # Concatenate: (batch_size, n, n, 2d)
        x_cat = torch.cat((x_left, x_right), dim=3)

        # Linear 연산 후 squeeze: (batch_size, n, n)
        att_adj = self.linear(x_cat).squeeze(-1)

        # sigmoid activation
        att_adj = F.sigmoid(att_adj)

        # adj에 attention mask 적용
        if adj.is_sparse:
            adj = adj.to_dense()

        att_adj = att_adj * adj
        return att_adj


class Manifold(object):
    """
    Abstract class to define operations on a manifold.
    """

    def __init__(self):
        super().__init__()
        self.eps = 10e-8

    def sqdist(self, p1, p2, c):
        """Squared distance between pairs of points."""
        raise NotImplementedError

    def egrad2rgrad(self, p, dp, c):
        """Converts Euclidean Gradient to Riemannian Gradients."""
        raise NotImplementedError

    def proj(self, p, c):
        """Projects point p on the manifold."""
        raise NotImplementedError

    def proj_tan(self, u, p, c):
        """Projects u on the tangent space of p."""
        raise NotImplementedError

    def proj_tan0(self, u, c):
        """Projects u on the tangent space of the origin."""
        raise NotImplementedError

    def expmap(self, u, p, c):
        """Exponential map of u at point p."""
        raise NotImplementedError

    def logmap(self, p1, p2, c):
        """Logarithmic map of point p1 at point p2."""
        raise NotImplementedError

    def expmap0(self, u, c):
        """Exponential map of u at the origin."""
        raise NotImplementedError

    def logmap0(self, p, c):
        """Logarithmic map of point p at the origin."""
        raise NotImplementedError

    def mobius_add(self, x, y, c, dim=-1):
        """Adds points x and y."""
        raise NotImplementedError

    def mobius_matvec(self, m, x, c):
        """Performs hyperboic martrix-vector multiplication."""
        raise NotImplementedError

    def init_weights(self, w, c, irange=1e-5):
        """Initializes random weigths on the manifold."""
        raise NotImplementedError

    def inner(self, p, c, u, v=None, keepdim=False):
        """Inner product for tangent vectors at point x."""
        raise NotImplementedError

    def ptransp(self, x, y, u, c):
        """Parallel transport of u from x to y."""
        raise NotImplementedError

    def ptransp0(self, x, u, c):
        """Parallel transport of u from the origin to y."""
        raise NotImplementedError


class Hyperboloid(Manifold):
    """
    Hyperboloid manifold class.

    We use the following convention: -x0^2 + x1^2 + ... + xd^2 = -K

    c = 1 / K is the hyperbolic curvature. 
    """

    def __init__(self):
        super(Hyperboloid, self).__init__()
        self.name = 'Hyperboloid'
        self.eps = {
            torch.float16: 1e-4,
            torch.float32: 1e-7, 
            torch.float64: 1e-15
        }
        self.min_norm = 1e-15
        self.max_norm = 1e6

    def minkowski_dot(self, x, y, keepdim=True):
        res = torch.sum(x * y, dim=-1) - 2 * x[..., 0] * y[..., 0] # 앞에서 다 포함시켜서 더하고 뒤에서 두 번 빼기
        if keepdim:
            res = res.view(res.shape + (1,))
        return res

    def minkowski_norm(self, u, keepdim=True):
        dot = self.minkowski_dot(u, u, keepdim=keepdim)
        return torch.sqrt(torch.clamp(dot, min=self.eps[u.dtype]))

    def sqdist(self, x, y, c):
        K = 1. / c
        prod = self.minkowski_dot(x, y)
        theta = torch.clamp(-prod / K, min=1.0 + self.eps[x.dtype], max=self.max_norm)
        sqdist = K * torch.acosh(theta) ** 2
        # clamp distance to avoid nans in Fermi-Dirac decoder
        # return torch.clamp(sqdist, max=50.0)
        return sqdist

    def proj(self, x, c):
        K = 1. / c
        d = x.size(-1) - 1  # 공간 차원 (Lorentz 공간에서 시간 차원 제외한 나머지)
        y = x.narrow(-1, 1, d)  # x의 1번 인덱스부터 끝까지 (즉, x_1, x_2, ..., x_d)
        y_sqnorm = torch.norm(y, p=2, dim=-1, keepdim=True) ** 2  # 공간 좌표의 유클리드 제곱 노름 ||y||^2
        mask = torch.ones_like(x)
        mask[..., 0] = 0  # 시간 좌표는 별도로 업데이트할 것이므로 0으로 설정
        vals = torch.zeros_like(x)
        vals[..., 0:1] = torch.sqrt(torch.clamp(K + y_sqnorm, min=self.eps[x.dtype]))  # 새로운 시간 좌표 계산
        return vals + mask * x

    def proj_tan(self, u, x, c):
        K = 1. / c
        d = x.size(-1) - 1
        ux = torch.sum(x.narrow(-1, 1, d) * u.narrow(-1, 1, d), dim=-1, keepdim=True)
        mask = torch.ones_like(u)
        mask[..., 0] = 0
        vals = torch.zeros_like(u)
        vals[..., 0:1] = ux / torch.clamp(x[..., 0:1], min=self.eps[x.dtype])
        return vals + mask * u

    def proj_tan0(self, u, c):
        # narrowed = u.narrow(-1, 0, 1)
        # vals = torch.zeros_like(u)
        # vals[:, 0:1] = narrowed
        # return u - vals
        projected_u = u.clone()
        projected_u[..., 0] = 0
        return projected_u

    def expmap(self, u, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        normu = self.minkowski_norm(u)
        normu = torch.clamp(normu, max=self.max_norm)
        theta = normu / sqrtK
        theta = torch.clamp(theta, min=self.min_norm)
        result = torch.cosh(theta) * x + torch.sinh(theta) * u / theta
        return self.proj(result, c)
        
    def logmap(self, x, y, c):
        K = 1. / c
        xy = torch.clamp(self.minkowski_dot(x, y) + K, max=-self.eps[x.dtype]) - K
        u = y + xy * x * c
        normu = self.minkowski_norm(u)
        normu = torch.clamp(normu, min=self.min_norm)
        dist = self.sqdist(x, y, c) ** 0.5
        result = dist * u / normu
        return self.proj_tan(result, x, c)

    def expmap0(self, u, c):
        K = 1. / c
        sqrtK = K ** 0.5
        d = u.size(-1) - 1
        # x = u.narrow(-1, 1, d).view(-1, d)
        x = u.narrow(-1, 1, d).view(*u.shape[:-1], d)
        x_norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        x_norm = torch.clamp(x_norm, min=self.min_norm)
        theta = x_norm / sqrtK
        res = torch.ones_like(u)
        res[..., 0:1] = sqrtK * torch.cosh(theta)
        # res[..., 1:] = sqrtK * torch.sinh(theta) * x / x_norm
        res[..., 1:] = sqrtK * torch.sinh(theta).expand_as(x) * x / x_norm  # expand_as(x) 추가

        return self.proj(res, c)

    def logmap0(self, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        d = x.size(-1) - 1
        # y = x.narrow(-1, 1, d).view(-1, d)
        y = x.narrow(-1, 1, d).view(*x.shape[:-1], d) 
        y_norm = torch.norm(y, p=2, dim=-1, keepdim=True)
        y_norm = torch.clamp(y_norm, min=self.min_norm)
        res = torch.zeros_like(x)
        theta = torch.clamp(x[..., 0:1] / sqrtK, min=1.0 + self.eps[x.dtype])
        # res[..., 1:] = sqrtK * torch.arcosh(theta) * y / y_norm
        # res[..., 1:] = sqrtK * torch.log(theta + torch.sqrt(theta**2 - 1)) * y / y_norm
        log_term = torch.log(theta + torch.sqrt(theta**2 - 1))  # shape: [..., 1]

        res[..., 1:] = sqrtK * log_term.expand_as(y) * y / y_norm
        return res

    def mobius_add(self, x, y, c):
        u = self.logmap0(y, c)
        v = self.ptransp0(x, u, c)
        return self.expmap(v, x, c)

    def mobius_matvec(self, m, x, c):
        u = self.logmap0(x, c)
        mu = u @ m.transpose(-1, -2)
        return self.expmap0(mu, c)

    def ptransp(self, x, y, u, c):
        logxy = self.logmap(x, y, c)
        logyx = self.logmap(y, x, c)
        sqdist = torch.clamp(self.sqdist(x, y, c), min=self.min_norm)
        alpha = self.minkowski_dot(logxy, u) / sqdist
        res = u - alpha * (logxy + logyx)
        return self.proj_tan(res, y, c)

    def ptransp0(self, x, u, c):
        K = 1. / c
        sqrtK = K ** 0.5
        x0 = x.narrow(-1, 0, 1)
        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)
        y_norm = torch.clamp(torch.norm(y, p=2, dim=-1, keepdim=True), min=self.min_norm)
        y_normalized = y / y_norm
        v = torch.ones_like(x)
        v[..., 0:1] = - y_norm 
        v[..., 1:] = (sqrtK - x0) * y_normalized
        alpha = torch.sum(y_normalized * u[..., 1:], dim=-1, keepdim=True) / sqrtK
        # res = u - alpha * v
        # alpha = alpha.unsqueeze(-2)
        res = u - alpha * v
        return self.proj_tan(res, x, c)

    def to_poincare(self, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        d = x.size(-1) - 1
        return sqrtK * x.narrow(-1, 1, d) / (x[..., 0:1] + sqrtK)

