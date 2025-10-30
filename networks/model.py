import torch
import torch.nn as nn
import torch.nn.functional as F

from network.utils import compute_euclid_distance, add_full_rrwp
from network.gnns import Cheb1, Cheb2, Hyperboloid, HyperbolicGraphConvolution, myCheb

class NodeFeatureFusionMinimal(nn.Module):
    """
    mode:
      - "minimal":  z = LN( x + gamma * W * decay_sum(p) )
      - "legacy":   기존 방식(p->lin, concat, lin)
    Args
      Dx: x 채널 수, K: RRWP hop 수, out_dim: 최종 채널
      alpha: 감쇠계수, learn_gamma: 전역 혼합계수 학습 여부
      stopgrad_p: p 브랜치 stop-grad (권장 True)
      position_dim: legacy용
    """
    def __init__(self, Dx, K, out_dim,
                 mode="minimal",
                 alpha=0.75,
                 learn_gamma=True,
                 gamma_init=0.5,
                 stopgrad_p=True,
                 position_dim=16):
        super().__init__()
        assert mode in ["minimal","origin", "sum"]
        self.mode = mode
        self.alpha = alpha
        self.stopgrad_p = stopgrad_p

        

        if mode == "minimal":
            self.ln_x   = nn.LayerNorm(Dx)
            self.ln_out = nn.LayerNorm(out_dim)
            self.p_proj = nn.Linear(1, Dx, bias=False)              # 아주 작은 선형
            self.out    = nn.Identity() if out_dim==Dx else nn.Linear(Dx, out_dim, bias=True)
            if learn_gamma:
                # 전역 스칼라 혼합계수 gamma \in (0,1)
                logit = torch.logit(torch.tensor(gamma_init, dtype=torch.float32).clamp(1e-4, 1-1e-4))
                self._gamma = nn.Parameter(logit)
            else:
                self.register_buffer("_gamma_const", torch.tensor(gamma_init, dtype=torch.float32))
                self._gamma = None
        elif mode == "origin":
            self.ln_rw   = nn.Linear(K, position_dim)               # 기존
            self.init_ln = nn.Linear(Dx + position_dim, out_dim)
        elif mode == 'sum':
            self.ln_rw = nn.Linear(K, Dx)
            self.ln_out = nn.LayerNorm(out_dim)

    def gamma(self):
        if self._gamma is None:
            return self._gamma_const
        return torch.sigmoid(self._gamma)

    def forward(self, x, p):
        # x: [B,N,Dx], p: [B,N,K]
        if self.stopgrad_p:
            p = p.detach()

        if self.mode == 'origin':
            po  = self.ln_rw(p)                                     # [B,N,pos_dim]
            xp  = torch.cat([x, po], dim=-1)
            z   = self.init_ln(xp)                                  # [B,N,out_dim]
            return z 
        elif self.mode == 'sum':
            po = self.ln_rw(p)
            z = x + po
            return self.ln_out(z)

class Spark(nn.Module):
    def __init__(self, config): #, hparams):
        super().__init__()
        
        self.L = config['layers']
        self.c = 1.0
        self.init_input = config['init_dim']
        self.hidden_dim = config['feature_dim']

        self.d = self.init_input+1
        self.h = self.hidden_dim+1
        self.rw = config['random_walk']
        self.position_dim = config['position_dim']

        self.learn_alpha = config['learn_alpha']
        self.fuse_mode = config['fuse_mode']

        if self.rw > 0:
            self.node_fusion = NodeFeatureFusionMinimal(Dx=self.init_input, K=self.rw,
                                                        out_dim=self.init_input, mode=self.fuse_mode,
                                                        position_dim=self.position_dim)  
        self.manifold = Hyperboloid()

        self.graph_f_euclidean = nn.ModuleList() 
        self.graph_f_hyperbolic = nn.ModuleList() 

        if self.L == 1:
            self.graph_f_euclidean.append(Cheb1(self.init_input, self.hidden_dim))
            # self.graph_f_euclidean.append(myChebConv(self.init_input, self.hidden_dim))
            self.graph_f_hyperbolic.append(HyperbolicGraphConvolution(self.manifold,self.d,self.h,self.c,self.c,0.0,act=nn.Mish(), use_bias=True, use_att=False, local_agg=False))
        else:
            self.graph_f_euclidean.append(Cheb1(self.init_input, self.hidden_dim))
            self.graph_f_hyperbolic.append(HyperbolicGraphConvolution(self.manifold,self.d,self.h,self.c,self.c,0.0,act=nn.Mish(), use_bias=True, use_att=False, local_agg=False))

            for l in range(self.L-1):
                self.graph_f_euclidean.append(Cheb1(self.hidden_dim, self.hidden_dim))
                self.graph_f_hyperbolic.append(HyperbolicGraphConvolution(self.manifold,self.h, self.h,self.c,self.c,0.0,act=nn.Mish(), use_bias=True, use_att=False, local_agg=False))
            
        self._temp = nn.Parameter(torch.tensor(config['temperature']))
        if self.learn_alpha:
            self._alpha_a = nn.Parameter(torch.tensor(0.0))
        else:
            self._alpha_a = config['alpha']

        self.node_g = Cheb2(self.init_input, self.init_input, self.hidden_dim, self.init_input)


    def temperature(self):
        return F.softplus(self._temp) + 1e-6
    
    def alpha_a(self):
        if self.learn_alpha:
            return torch.sigmoid(self._alpha_a)
        else:
            return self._alpha_a

    def forward(self, x, edge_weight, edges):
        b,n,d = x.shape
        
        if self.rw > 0:
            p = add_full_rrwp(torch.abs(edge_weight), edges, self.rw)
            x_p = self.node_fusion(x, p)
        else:
            x_p = x

        graph_x_e = x_p

        o = torch.zeros_like(x_p)
        x_h = torch.cat([o[..., 0:1], x_p], dim=-1)

        graph_x_tan = self.manifold.proj_tan0(x_h, c=self.c)
        graph_x_h = self.manifold.expmap0(graph_x_tan, c=self.c)

        # lprobslist = []
        dist_e_list = []
        dist_h_list = []
        a = self.alpha_a() 

        for i, (f_e, f_h) in enumerate(zip(self.graph_f_euclidean, self.graph_f_hyperbolic)):
            if a == 1.0:
                with torch.no_grad():
                    _ = f_e(graph_x_e, edges)  # forward도 막기 위해 실행하긴 함
                graph_x_e = torch.zeros((b,n,self.h-1))

                if not torch.isfinite(graph_x_h).all():
                    print("NaN/Inf in graph_x_h after Hyp layer"); raise RuntimeError

                graph_x_h_1 = graph_x_h.unsqueeze(2) #.expand(b, n, n, self.h)  # [batch, node, node, feature]
                graph_x_h_2 = graph_x_h.unsqueeze(1) #.expand(b, n, n, self.h)
            
                dist_h = self.manifold.sqdist(graph_x_h_1, graph_x_h_2, c=self.c).squeeze() # 0~(2.3842e-07, 47684e-07)~(3,5)~50 평균은 10 std 16
                diag_mask = torch.eye(n, device=dist_h.device).bool().unsqueeze(0).expand(b, -1, -1)
                off_diag_mask = ~diag_mask

                h_off_diag_vals = dist_h[off_diag_mask].view(b, -1)
                h_min, h_max = h_off_diag_vals.min(dim=1)[0].view(b,1,1), h_off_diag_vals.max(dim=1)[0].view(b,1,1)

                dist_h_scaled = (dist_h - h_min) / (h_max - h_min + 1e-8)
                dist_e_scaled = torch.zeros(dist_h_scaled.shape)
                total_dist = dist_h_scaled

                dist_h_list.append(dist_h)

            elif a == 0.0:
                with torch.no_grad():
                    _ = f_h(graph_x_h, edges)
                graph_x_h = torch.zeros((b,n,self.h))

                dist_e = compute_euclid_distance(graph_x_e) # 0~(1.1~2.1)~(4,5,6,7,8,9)~11 평균 3.2697, std 0.9945
                
                diag_mask = torch.eye(n, device=dist_e.device).bool().unsqueeze(0).expand(b, -1, -1)
                off_diag_mask = ~diag_mask

                e_off_diag_vals = dist_e[off_diag_mask].view(b, -1)
                e_min, e_max = e_off_diag_vals.min(dim=1)[0].view(b,1,1), e_off_diag_vals.max(dim=1)[0].view(b,1,1)
                
                dist_e_scaled = (dist_e - e_min) / (e_max - e_min + 1e-8)
                dist_h_scaled = torch.zeros(dist_e_scaled.shape)
                total_dist = dist_e_scaled
                
                dist_e_list.append(dist_e)
                
            else:
                graph_x_e = f_e(graph_x_e, edges)
                graph_x_h = f_h(graph_x_h, edges)

           
                # DualGSL.forward 안에 한 번만 넣어서 확인
                if not torch.isfinite(graph_x_e).all():
                    print("NaN in graph_x_e after GCN"); raise RuntimeError
                if not torch.isfinite(graph_x_h).all():
                    print("NaN/Inf in graph_x_h after Hyp layer"); raise RuntimeError


                dist_e = compute_euclid_distance(graph_x_e) # 0~(1.1~2.1)~(4,5,6,7,8,9)~11 평균 3.2697, std 0.9945
                
                graph_x_h_1 = graph_x_h.unsqueeze(2) #.expand(b, n, n, self.h)  # [batch, node, node, feature]
                graph_x_h_2 = graph_x_h.unsqueeze(1) #.expand(b, n, n, self.h)

                dist_h = self.manifold.sqdist(graph_x_h_1, graph_x_h_2, c=self.c).squeeze(-1) # 0~(2.3842e-07, 47684e-07)~(3,5)~50 평균은 10 std 16
                dist_h = torch.sqrt(dist_h)

                diag_mask = torch.eye(n, device=dist_e.device).bool().unsqueeze(0).expand(b, n, n)
                
                # 대칭 보정 및 대각 0 고정
                dist_e = 0.5 * (dist_e + dist_e.transpose(1, 2))
                dist_h = 0.5 * (dist_h + dist_h.transpose(1, 2))
                dist_e = dist_e.masked_fill(diag_mask, 0.0)
                dist_h = dist_h.masked_fill(diag_mask, 0.0)

                # 배치별 off-diagonal min/max 계산
                # min은 대각을 +inf로 채워서, max는 대각을 -inf로 채워서 구하면 깔끔
                BIG = torch.finfo(dist_e.dtype).max
                e_min = dist_e.masked_fill(diag_mask, BIG).amin(dim=(1, 2), keepdim=True)   # [B,1,1]
                e_max = dist_e.masked_fill(diag_mask, -BIG).amax(dim=(1, 2), keepdim=True)  # [B,1,1]
                h_min = dist_h.masked_fill(diag_mask, BIG).amin(dim=(1, 2), keepdim=True)
                h_max = dist_h.masked_fill(diag_mask, -BIG).amax(dim=(1, 2), keepdim=True)

                # 분모 안정화
                e_den = torch.clamp(e_max - e_min, min=1e-8)
                h_den = torch.clamp(h_max - h_min, min=1e-8)

                dist_e_scaled = (dist_e - e_min) / e_den
                dist_h_scaled = (dist_h - h_min) / h_den

                # 혹시 모를 수치잡음 정리
                dist_e_scaled = dist_e_scaled.masked_fill(diag_mask, 0.0).clamp(0.0, 1.0)
                dist_h_scaled = dist_h_scaled.masked_fill(diag_mask, 0.0).clamp(0.0, 1.0)

                total_dist = (1 - a) * (dist_e_scaled**2) + a * (dist_h_scaled**2)
                dist_e_list.append(dist_e_scaled)
                dist_h_list.append(dist_h_scaled)

            # percentile
            percentile = 0.50
            total_off = total_dist[off_diag_mask].view(b, -1)
            percentile_thresh = torch.quantile(total_off, percentile, dim=1).view(b,1,1)
            edges = torch.sigmoid(self.temperature() * (percentile_thresh - total_dist))
            init_edges = torch.sigmoid(self.temperature() * (percentile_thresh - total_dist))
            
            init_edges_off_diag_vals = init_edges[off_diag_mask].view(b, -1)
            batch_min, batch_max = init_edges_off_diag_vals.min(dim=1)[0].view(b,1,1), init_edges_off_diag_vals.max(dim=1)[0].view(b,1,1)
            
            edges = (init_edges - batch_min) / (batch_max - batch_min + 1e-8)
            edges = edges.masked_fill(diag_mask, 1.0)
            self.edges=edges
        
        graph_x_h_e = self.manifold.logmap0(graph_x_h, self.c)
        graph_x_h_e = self.manifold.proj_tan0(graph_x_h_e, self.c)

        logits = self.node_g(x_p, edges)

        return logits
    

