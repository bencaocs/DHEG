import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F 
from EGNN import EGNN
import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import pandas as pd
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

# 构建LeakyReLU函数
leakyRelu = nn.LeakyReLU(0.1) 


def normalize_tensor(mx, eqvar=None):
    """
    Row-normalize sparse matrix
    """
    rowsum = torch.sum(mx, 1)
    if eqvar:
        r_inv = torch.pow(rowsum, -1 / eqvar).flatten()
        r_inv[torch.isinf(r_inv)] = 0.0
        r_mat_inv = torch.diag(r_inv)
        mx = torch.mm(r_mat_inv, mx)
        return mx

    else:
        r_inv = torch.pow(rowsum, -1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.0
        r_mat_inv = torch.diag(r_inv)
        mx = torch.mm(r_mat_inv, mx)
        return mx

device = f"cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)


######
class ImprovedFeatureWeighting(nn.Module):
    def __init__(self, feature_dims, hidden_dim=32, init_scale=1.0):
        super().__init__()
        self.feature_dims = feature_dims
        self.num_features = len(feature_dims)
        self.total_dim = sum(feature_dims)
        self.init_scale = init_scale
        
        # 注意力机制 - 更简单的设计
        self.attention = nn.Sequential(
            nn.Linear(self.total_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # 添加少量dropout防止过拟合
            nn.Linear(hidden_dim, self.num_features),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch_size, seq_len, total_dim = x.shape
        

        global_feat = x.mean(dim=1)
        importance_scores = self.attention(global_feat)
        
  
        weighted_features = []
        start_idx = 0
        for i, dim in enumerate(self.feature_dims):
            end_idx = start_idx + dim
            feature_slice = x[:, :, start_idx:end_idx]
            

            weight = importance_scores[:, i].unsqueeze(1).unsqueeze(2)
            weighted_feature = feature_slice * (self.init_scale + weight)
            
            weighted_features.append(weighted_feature)
            start_idx = end_idx
        
        weighted_x = torch.cat(weighted_features, dim=-1)
        
        return weighted_x, importance_scores


########################################################################################3



class Subgraphnet(nn.Module):
    def __init__(self, ks, in_dim, out_dim, dim, act=F.hardtanh, drop_p=0.0):
        super(Subgraphnet, self).__init__()
        self.ks = ks

        self.act = act
        self.down_gcns = nn.ModuleList()
        self.up_gcns = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.unpools = nn.ModuleList()

        self.subgraph_gcns_1 = nn.ModuleList()
        # self.subgraph_gcns_2 = nn.ModuleList()
        
        self.norm = nn.LayerNorm(dim)
        self.l_n = len(ks)
        self.drop_p = drop_p
        self.fusion_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        self.skip_conv = nn.Linear(dim, dim)  
        self.egnn_gate = nn.Linear(dim * 2, dim)  

        #######################  features group sort #######################

        # feature_dims = [20, 20, 14, 7, 2]  # pssm, hmm, dssp, atom, dist_abs, dist_rel
        # self.feature_reorder = DynamicFeatureReorder(feature_dims, hidden_dim=64)
        
        # self.feature_projection = nn.Linear(sum(feature_dims), dim)

        # self.register_buffer('feature_weights', torch.zeros(len(feature_dims)))
        # self.register_buffer('weight_count', torch.tensor(0))


        # feature_dims = [20, 20, 14, 7, 2] 
        
        # self.feature_reorder = SimplifiedFeatureReorder(feature_dims, hidden_dim=32)
        

        # self.residual_scale = nn.Parameter(torch.tensor(1.0))
        

        # self.register_buffer('feature_weights', torch.zeros(len(feature_dims)))
        # self.register_buffer('weight_count', torch.tensor(0))

        #######

        feature_dims = [20, 20, 14, 7, 2]  #[20, 20, 14, 7, 2]
        

        self.feature_weighting = ImprovedFeatureWeighting(feature_dims, hidden_dim=64, init_scale=0.8)
        

        self.residual_alpha = nn.Parameter(torch.tensor(0.3))  #0.3
        

        self.register_buffer('feature_weights', torch.zeros(len(feature_dims)))
        self.register_buffer('weight_count', torch.tensor(0))

        #####################################################




        for j in range(6):
            # print(j)
            self.subgraph_gcns_1.append(EGNN(dim = in_dim,edge_dim=1, m_dim =17))# ,dropout=self.drop_p)) # 16 20
            # self.subgraph_gcns_1.append(GCNConv(in_dim, in_dim))
            # self.subgraph_gcns_2.append(GCNConv(in_dim, in_dim))
            # self.subgraph_gcns_1.append(GATConv(in_dim,in_dim,concat=False,heads=3,dropout=0.5))

        for i in range(self.l_n):
           
            self.pools.append(Pool(ks[i], dim, drop_p))
            self.unpools.append(Unpool(dim, dim, drop_p))

    def readout(self, hs):
        h_max = torch.Tensor([torch.max(h, 0)[0] for h in hs])
        # h_sum = [torch.sum(h, 0) for h in hs]
        # h_mean = [torch.mean(h, 0) for h in hs]
        # h = torch.cat(h_max + h_sum + h_mean)
        return h_max

    # 注意力子图 egnn 模型
    def forward(self, feat, coor, edge , ep):


        ##############################################################################################################
        original_feat = feat
        
        


 
        weighted_feat, importance_scores = self.feature_weighting(feat)

	#figure 
        # scores_np = importance_scores.detach().cpu().numpy().squeeze()
        # data_row = {
        #     'epoch': ep,
        #     'source_1': scores_np[0],
        #     'source_2': scores_np[1], 
        #     'source_3': scores_np[2],
        #     'source_4': scores_np[3],
        #     'source_5': scores_np[4],
        #     'sum': scores_np.sum(),
        #     'mean': scores_np.mean()
        # }
        # df = pd.DataFrame([data_row])

        # if not os.path.exists('training_importance.csv'):
        #     df.to_csv('training_importance.csv', index=False)
        # else:
        #     df.to_csv('training_importance.csv', mode='a', header=False, index=False)
        # feat_np = original_feat.cpu().detach().numpy().squeeze(0)  # [79, 63]
        # scores_np = importance_scores.cpu().detach().numpy().squeeze(0)  # [5]
        
        # feature_dims = [20, 20, 14, 7, 2]
        # source_strengths = []
        # start_idx = 0
        # for dim in feature_dims:
        #     end_idx = start_idx + dim
        #     source_feat = feat_np[:, start_idx:end_idx]
        #     source_strengths.append(np.mean(np.abs(source_feat)))
        #     start_idx = end_idx
        
        # data_row = {
        #     'epoch': ep,

        #     'imp_source1': scores_np[0], 'imp_source2': scores_np[1], 'imp_source3': scores_np[2],
        #     'imp_source4': scores_np[3], 'imp_source5': scores_np[4],
        #     'imp_sum': scores_np.sum(), 'imp_mean': scores_np.mean(),
  
        #     'str_source1': source_strengths[0], 'str_source2': source_strengths[1],
        #     'str_source3': source_strengths[2], 'str_source4': source_strengths[3],
        #     'str_source5': source_strengths[4],
        #     'str_sum': sum(source_strengths), 'str_mean': np.mean(source_strengths)
        # }
        
  
        # with open('training_data.csv', 'a') as f:
        #     if ep == 0:  # 
        #         f.write(','.join(data_row.keys()) + '\n')
        #     f.write(','.join(map(str, data_row.values())) + '\n')

        mixed_feat = (1 - self.residual_alpha) * original_feat + self.residual_alpha * weighted_feat
        
        

        if self.training:
            self.feature_weights = (self.feature_weights * self.weight_count + 
                                  importance_scores.mean(dim=0)) / (self.weight_count + 1)
            self.weight_count += 1
        feat = mixed_feat
        h=feat.squeeze(0)
        # ##############################################################################################################

        # adj_ms = []
        # indices_list = []
        # down_outs = []
        # hs = []
        # org_h = feat[0,:,:]
        # org_g = edge[0,:,:,0]
        # org_c = coor[0,:,:]
        
        # for i in range(self.l_n):
        #     g = org_g
        #     h = org_h
        #     c = org_c
        #     if self.ks[i] != 1:
        #         # print('out')
        #         g, h, idx = self.pools[i](g, h, ep)
        #         # print(idx)
        #         c = c[idx,:]


        #     feat1 = h.unsqueeze(0)
        #     edge1 = g.unsqueeze(0)
        #     edge1 = edge1.unsqueeze(edge1.dim())
        #     coor1 = c.unsqueeze(0)
        #     h = feat1
            
        #     for j in range(6):
        #         # h = h +  self.subgraph_gcns_1[j](h, coor1, edge1,ep)
        #         # h = torch.relu(h)

        #         # h_temp, coor1 =  self.subgraph_gcns_1[j](h, coor1, edge1,ep)
        #         # h = torch.relu(h + h_temp)
        #         ########################################################################################################
        #         h_temp, coor1 = self.subgraph_gcns_1[j](h, coor1, edge1, ep)
        #         h_gate = torch.sigmoid(self.egnn_gate(torch.cat([h, h_temp], dim=-1)))  
        #         h = h_gate * h + (1 - h_gate) * h_temp

        #         if i > 0:
        #             h = h + self.skip_conv(feat1)  

        #         #######################################################################################################


        #         # h = torch.sigmoid(h)
        #         # h = F.hardtanh(h)

        #     h = h[0,:,:]

        #     if self.ks[i] != 1:
        #         # print('done')
        #         g, h = self.unpools[i](org_g, h, org_h, idx)
        #         # print(idx)
        #         # exit()
        #     hs.append(h)
        
        # # for h_i in hs:
        # #     # print(h_i)
        # #     # print(len[i > 0 for i in h_i])
        # #     h = torch.max(h,h_i)

        # if len(hs) > 1:
        #     h_stack = torch.stack(hs, dim=1)  # [batch, n_levels, dim]
        #     h_avg = h_stack.mean(dim=1)
        #     h_max = h_stack.max(dim=1).values
        #     gate = self.fusion_gate(torch.cat([h_avg, h_max], dim=-1))
        #     h = gate * h_avg + (1 - gate) * h_max
        # else:
        #     h = hs[0]     

        return h

    
class Pool(nn.Module):

    def __init__(self, k, in_dim, p=0.3):
        super(Pool, self).__init__()
        self.k = k
        self.max_heads = 4
        self.heads = max([h for h in range(1, self.max_heads) if in_dim % h == 0])
        assert self.heads > 0, f"No valid heads for in_dim {in_dim}"
        

        self.qkv = nn.Linear(in_dim, in_dim * 3)  # 合并QKV投影
        self.dropout = nn.Dropout(p=p)
      
        self.degree_weight = nn.Parameter(torch.tensor(0.1))
        self.score_proj = nn.Linear(in_dim, 1)

    def forward(self, g, h, ep):
        n_nodes = h.size(0)

        qkv = self.qkv(h).chunk(3, dim=-1)  # [Q, K, V]
        q, k, v = [x.view(n_nodes, self.heads, -1) for x in qkv]  # [n, h, d_h]
        
        attn = (q @ k.transpose(-2, -1)) / (q.size(-1) ** 0.5)  # [n, h, n]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        

        out = attn @ v  # [n, h, d_h]
        out = out.transpose(1, 2).reshape(n_nodes, -1)  # [n, d]
        

        scores = self.score_proj(out).squeeze(-1)  # [n]

        return top_k_graph(scores, g, h, self.k, ep)



class Unpool(nn.Module):

    def __init__(self, *args):
        super(Unpool, self).__init__()

    def forward(self, g, h, pre_h, idx):
        # print(h[5])
        new_h = h.new_zeros([g.shape[0], h.shape[1]])
        
        new_h[idx] = h

        return g, new_h


def top_k_graph(scores, g, h, k, ep):
    num_nodes = g.shape[0]
    values, idx = torch.topk(scores, max(2, int(k*num_nodes)))
    # print(idx)
    new_h = h[idx, :]
    values = torch.unsqueeze(values, -1)
    new_h = torch.mul(new_h, values)#

    un_g = g.bool().float()
    un_g = torch.matmul(un_g, un_g).bool().float()#
    un_g = un_g[idx, :]
    un_g = un_g[:, idx]
    # g = norm_g(un_g)#
    # print(g.shape)
    # print(un_g.shape)
    # exit()
    g = un_g
    return g, new_h, idx


def norm_g(g):
    degrees = torch.sum(g, 1)
    g = g / degrees
    return g


class Initializer(object):

    @classmethod
    def _glorot_uniform(cls, w):
        if len(w.size()) == 2:
            fan_in, fan_out = w.size()
        elif len(w.size()) == 3:
            fan_in = w.size()[1] * w.size()[2]
            fan_out = w.size()[0] * w.size()[2]
        else:
            fan_in = np.prod(w.size())
            fan_out = np.prod(w.size())
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        w.uniform_(-limit, limit)

    @classmethod
    def _param_init(cls, m):
        if isinstance(m, nn.parameter.Parameter):
            cls._glorot_uniform(m.data)
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()
            cls._glorot_uniform(m.weight.data)

    @classmethod
    def weights_init(cls, m):
        for p in m.modules():
            if isinstance(p, nn.ParameterList):
                for pp in p:
                    cls._param_init(pp)
            else:
                cls._param_init(p)

        for name, p in m.named_parameters():
            if '.' not in name:
                cls._param_init(p)