import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F 
# from torch_geometric.nn import MessagePassing, GCNConv,GATConv
# from torch_geometric.utils import add_self_loops, degree
# from torch_geometric.utils import dense_to_sparse
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
# 创建 t-SNE 对象并设置参数
tsne = TSNE(n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=1000)


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

#######################新加 ##################3
# class DynamicFeatureReorder(nn.Module):
#     def __init__(self, feature_dims, hidden_dim=64):
#         super().__init__()
#         self.feature_dims = feature_dims
#         self.num_features = len(feature_dims)
#         self.total_dim = sum(feature_dims)
        
#         # 注意力机制学习特征重要性
#         self.attention = nn.Sequential(
#             nn.Linear(self.total_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim, self.num_features),
#             nn.Softmax(dim=-1)
#         )
        
#     def forward(self, x):
#         batch_size, seq_len, total_dim = x.shape
        
#         # 验证输入维度
#         if total_dim != self.total_dim:
#             raise ValueError(f"Input dimension {total_dim} doesn't match expected total dimension {self.total_dim}")
        
#         # 计算注意力权重（使用全局平均池化）
#         global_feat = x.mean(dim=1)  # [batch, total_dim]
#         attn_weights = self.attention(global_feat)  # [batch, num_features]
        
#         # 分割特征
#         features = []
#         start_idx = 0
#         for dim in self.feature_dims:
#             end_idx = start_idx + dim
#             feature_slice = x[:, :, start_idx:end_idx]
#             features.append(feature_slice)
#             start_idx = end_idx
        
#         # 根据注意力权重重新排序特征
#         _, indices = torch.sort(attn_weights, dim=-1, descending=True)
        
#         # 动态重排序（对每个样本单独处理）
#         reordered_features = []
#         for b in range(batch_size):
#             batch_feats = []
#             for idx in indices[b]:
#                 if idx < len(features):  # 安全检查
#                     batch_feats.append(features[idx][b])
#                 else:
#                     # 如果索引越界，使用第一个特征
#                     batch_feats.append(features[0][b])
#             # 拼接该样本的所有特征
#             reordered_batch = torch.cat(batch_feats, dim=-1)
#             reordered_features.append(reordered_batch.unsqueeze(0))
        
#         reordered_x = torch.cat(reordered_features, dim=0)
        
#         return reordered_x, attn_weights


# class SimplifiedFeatureReorder(nn.Module):
#     def __init__(self, feature_dims, hidden_dim=32):
#         super().__init__()
#         self.feature_dims = feature_dims
#         self.num_features = len(feature_dims)
#         self.total_dim = sum(feature_dims)
        
#         # 更简单的注意力机制
#         self.attention = nn.Sequential(
#             nn.Linear(self.total_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, self.num_features),
#             nn.Sigmoid()  # 改用Sigmoid而不是Softmax
#         )
        
#     def forward(self, x):
#         batch_size, seq_len, total_dim = x.shape
        
#         # 全局平均池化
#         global_feat = x.mean(dim=1)
#         # 计算特征重要性分数（0-1之间）
#         importance_scores = self.attention(global_feat)
        
#         # 分割特征并加权（不改变顺序，只调整重要性）
#         weighted_features = []
#         start_idx = 0
#         for i, dim in enumerate(self.feature_dims):
#             end_idx = start_idx + dim
#             feature_slice = x[:, :, start_idx:end_idx]
#             # 应用重要性权重
#             weight = importance_scores[:, i].unsqueeze(1).unsqueeze(2)
#             weighted_feature = feature_slice * weight
#             weighted_features.append(weighted_feature)
#             start_idx = end_idx
        
#         # 保持原始顺序拼接，但特征已经加权
#         reordered_x = torch.cat(weighted_features, dim=-1)
        
#         return reordered_x, importance_scores
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
        
        # 使用多种统计信息
        global_feat = x.mean(dim=1)
        # global_std = x.std(dim=1)
        # global_feat = torch.cat([global_mean, global_std], dim=1)  # 结合均值和标准差
        
        # 计算特征重要性权重
        importance_scores = self.attention(global_feat)
        
        # 分割特征并应用权重（保持原始顺序）
        weighted_features = []
        start_idx = 0
        for i, dim in enumerate(self.feature_dims):
            end_idx = start_idx + dim
            feature_slice = x[:, :, start_idx:end_idx]
            
            # 正确的权重应用方式：在基准值基础上调整
            weight = importance_scores[:, i].unsqueeze(1).unsqueeze(2)
            weighted_feature = feature_slice * (self.init_scale + weight)
            
            weighted_features.append(weighted_feature)
            start_idx = end_idx
        
        # 保持原始顺序拼接
        weighted_x = torch.cat(weighted_features, dim=-1)
        
        return weighted_x, importance_scores


########################################################################################3



class Subgraphnet(nn.Module):
    def __init__(self, ks, in_dim, out_dim, dim, act=F.hardtanh, drop_p=0.0):
        super(Subgraphnet, self).__init__()
        self.ks = ks
        # u_gcn
        # self.bottom_gcn = u_GCN(dim, dim, act, drop_p) 
        
        # acm gcn 
        # self.bottom_gcn = GraphConvolution(dim,dim)

        # egcn
        # self.bottom_gcn = EGNN(dim = in_dim,edge_dim=1)
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
        self.skip_conv = nn.Linear(dim, dim)  # 简单的线性跳跃连接
        self.egnn_gate = nn.Linear(dim * 2, dim)  # 每层EGNN一个gate

        ####################### 添加 #######################

        # feature_dims = [20, 20, 14, 7, 2]  # pssm, hmm, dssp, atom, dist_abs, dist_rel
        # self.feature_reorder = DynamicFeatureReorder(feature_dims, hidden_dim=64)
        
        # # 新增：特征维度变换层（将重排序后的特征映射到目标维度）
        # self.feature_projection = nn.Linear(sum(feature_dims), dim)

        # # 注册缓冲区来存储特征权重（用于监控）
        # self.register_buffer('feature_weights', torch.zeros(len(feature_dims)))
        # self.register_buffer('weight_count', torch.tensor(0))


        #######
        # # 修正特征维度（根据您的实际特征）
        # feature_dims = [20, 20, 14, 7, 2]  # 请确认这些维度对应哪些特征
        
        # # 使用简化版的特征重排序
        # self.feature_reorder = SimplifiedFeatureReorder(feature_dims, hidden_dim=32)
        
        # # 可选：添加残差连接，确保不会比原始拼接差
        # self.residual_scale = nn.Parameter(torch.tensor(1.0))
        
        # # 注册监控缓冲区
        # self.register_buffer('feature_weights', torch.zeros(len(feature_dims)))
        # self.register_buffer('weight_count', torch.tensor(0))

        #######

        feature_dims = [20, 20, 14, 7, 2]  #[20, 20, 14, 7, 2]
        
        # 使用改进的版本
        self.feature_weighting = ImprovedFeatureWeighting(feature_dims, hidden_dim=64, init_scale=0.8)
        
        # 可学习的残差权重（初始值较小）
        self.residual_alpha = nn.Parameter(torch.tensor(0.3))  #0.3
        
        # 监控
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




        ##############################新加###################
        # # 动态特征重排序
        # reordered_feat, attn_weights = self.feature_reorder(feat)
        
        # # 更新特征权重统计（用于监控）
        # if self.training:
        #     self.feature_weights = (self.feature_weights * self.weight_count + attn_weights.mean(dim=0)) / (self.weight_count + 1)
        #     self.weight_count += 1
    
        # projected_feat = self.feature_projection(reordered_feat)

        # feat = projected_feat

        ####

        # original_feat = feat
        
        # # 特征重排序（加权）
        # reordered_feat, importance_scores = self.feature_reorder(feat)
        
        # # 更新监控
        # if self.training:
        #     self.feature_weights = (self.feature_weights * self.weight_count + 
        #                           importance_scores.mean(dim=0)) / (self.weight_count + 1)
        #     self.weight_count += 1
        
        # # 残差连接：混合原始特征和重排序特征
        # mixed_feat = original_feat + self.residual_scale * reordered_feat

        # feat = mixed_feat

        ##############################################################################################################
        original_feat = feat
        
        


        # 应用特征加权
        weighted_feat, importance_scores = self.feature_weighting(feat)

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
        
        # # 计算特征源强度
        # feature_dims = [20, 20, 14, 7, 2]
        # source_strengths = []
        # start_idx = 0
        # for dim in feature_dims:
        #     end_idx = start_idx + dim
        #     source_feat = feat_np[:, start_idx:end_idx]
        #     source_strengths.append(np.mean(np.abs(source_feat)))
        #     start_idx = end_idx
        
        # # 创建数据行
        # data_row = {
        #     'epoch': ep,
        #     # 重要性权重
        #     'imp_source1': scores_np[0], 'imp_source2': scores_np[1], 'imp_source3': scores_np[2],
        #     'imp_source4': scores_np[3], 'imp_source5': scores_np[4],
        #     'imp_sum': scores_np.sum(), 'imp_mean': scores_np.mean(),
        #     # 特征源强度  
        #     'str_source1': source_strengths[0], 'str_source2': source_strengths[1],
        #     'str_source3': source_strengths[2], 'str_source4': source_strengths[3],
        #     'str_source5': source_strengths[4],
        #     'str_sum': sum(source_strengths), 'str_mean': np.mean(source_strengths)
        # }
        
        # # 保存到CSV（简单版）
        # with open('training_data.csv', 'a') as f:
        #     if ep == 0:  # 第一行写表头
        #         f.write(','.join(data_row.keys()) + '\n')
        #     f.write(','.join(map(str, data_row.values())) + '\n')

        # 残差混合：大部分保持原始特征，小部分使用加权特征
        mixed_feat = (1 - self.residual_alpha) * original_feat + self.residual_alpha * weighted_feat
        
        
        # 更新监控
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
        #         h_gate = torch.sigmoid(self.egnn_gate(torch.cat([h, h_temp], dim=-1)))  # 新增egnn_gate层
        #         h = h_gate * h + (1 - h_gate) * h_temp

        #         if i > 0:
        #             h = h + self.skip_conv(feat1)  # 需要新增skip_conv层

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

# class Subgraphnetmul(nn.Module):
#     def __init__(self, ks, in_dim, out_dim, dim, act=F.hardtanh, drop_p=0.0):
#         super(Subgraphnetmul, self).__init__()
#         self.ks = ks
#         # u_gcn
#         # self.bottom_gcn = u_GCN(dim, dim, act, drop_p) 
        
#         # acm gcn 
#         # self.bottom_gcn = GraphConvolution(dim,dim)

#         # egcn
#         # self.bottom_gcn = EGNN(dim = in_dim,edge_dim=1)
#         self.act = act
#         self.down_gcns = nn.ModuleList()
#         self.up_gcns = nn.ModuleList()
#         self.pools = nn.ModuleList()
#         self.unpools = nn.ModuleList()
#         self.graph_hidden_layer_output_dims=None
#         self.atom_layers = nn.ModuleList()
#         self.graph_layers = nn.ModuleList()
#         # self.subgraph_gcns_2 = nn.ModuleList()
        
#         self.norm = nn.LayerNorm(dim)
#         self.l_n = len(ks)
#         self.drop_p = drop_p
#         self.fusion_gate = nn.Sequential(
#             nn.Linear(dim * 2, dim),
#             nn.Sigmoid()
#         )
#         self.skip_conv = nn.Linear(dim, dim)  # 简单的线性跳跃连接
#         self.egnn_gate = nn.Linear(dim * 2, dim)  # 每层EGNN一个gate

#         if self.graph_hidden_layer_output_dims == None: self.graph_hidden_layer_output_dims = [in_dim]
#         # if linear_hidden_layer_output_dims == None: linear_hidden_layer_output_dims = []
#         # atom layers
#         for j in range(6):
#             # print(j)
#             self.atom_layers.append(EGNN(dim = in_dim,edge_dim=1, m_dim =17))# ,dropout=self.drop_p)) # 16 20
#             # self.subgraph_gcns_1.append(GCNConv(in_dim, in_dim))
#             # self.subgraph_gcns_2.append(GCNConv(in_dim, in_dim))
#             # self.subgraph_gcns_1.append(GATConv(in_dim,in_dim,concat=False,heads=3,dropout=0.5))
#         # graph layers
#         for k in [in_dim]*1:
#             self.graph_layers.append(EGNN(dim = in_dim,
#                                           edge_dim = 1,
#                                           m_dim = 17))



#         for i in range(self.l_n):
           
#             self.pools.append(Pool(ks[i], dim, drop_p))
#             self.unpools.append(Unpool(dim, dim, drop_p))

#     def readout(self, hs):
#         h_max = torch.Tensor([torch.max(h, 0)[0] for h in hs])
#         # h_sum = [torch.sum(h, 0) for h in hs]
#         # h_mean = [torch.mean(h, 0) for h in hs]
#         # h = torch.cat(h_max + h_sum + h_mean)
#         return h_max

#     # 注意力子图 egnn 模型
#     def forward(self, feat, coor, edge , ep):
    
#         adj_ms = []
#         indices_list = []
#         down_outs = []
#         hs = []
#         org_h = feat[0,:,:]
#         org_g = edge[0,:,:,0]
#         org_c = coor[0,:,:]
        
#         for i in range(self.l_n):
#             g = org_g
#             h = org_h
#             c = org_c
#             if self.ks[i] != 1:
#                 # print('out')
#                 g, h, idx = self.pools[i](g, h, ep)
#                 # print(idx)
#                 c = c[idx,:]
            
#             # egnn
#             feat1 = h.unsqueeze(0)
#             edge1 = g.unsqueeze(0)
#             edge1 = edge1.unsqueeze(edge1.dim())
#             coor1 = c.unsqueeze(0)
#             h = feat1
            
#             for j in range(6):
#                 # h = h +  self.subgraph_gcns_1[j](h, coor1, edge1,ep)
#                 # h = torch.relu(h)

#                 # h_temp, coor1 =  self.subgraph_gcns_1[j](h, coor1, edge1,ep)
#                 # h = torch.relu(h + h_temp)

#                 h_temp, coor1 = self.atom_layers[j](h, coor1, edge1, ep)
#                 h_gate = torch.sigmoid(self.egnn_gate(torch.cat([h, h_temp], dim=-1)))  # 新增egnn_gate层
#                 h = h_gate * h + (1 - h_gate) * h_temp

#                 if i > 0:
#                     h = h + self.skip_conv(feat1)  # 需要新增skip_conv层

#             for layer in self.graph_layers:  
#                 h_temp, coors1 = layer(h, coor1, edge1, ep)
#                 # feats = feats * 0.8 + torch.relu(feats)*0.2
#                 h= F.hardtanh(h_temp)                

#             h = h[0,:,:]

#             if self.ks[i] != 1:
#                 # print('done')
#                 g, h = self.unpools[i](org_g, h, org_h, idx)
#                 # print(idx)
#                 # exit()
#             hs.append(h)
        

#         if len(hs) > 1:
#             h_stack = torch.stack(hs, dim=1)  # [batch, n_levels, dim]
#             h_avg = h_stack.mean(dim=1)
#             h_max = h_stack.max(dim=1).values
#             gate = self.fusion_gate(torch.cat([h_avg, h_max], dim=-1))
#             h = gate * h_avg + (1 - gate) * h_max
#         else:
#             h = hs[0]     
        
#         return h
    
class GraphUnet(nn.Module):
    def __init__(self, ks, in_dim, out_dim, dim, act=F.hardtanh, drop_p=0.0):
        super(GraphUnet, self).__init__()
        self.ks = ks
        # u_gcn
        # self.bottom_gcn = u_GCN(dim, dim, act, drop_p) 
        
        # acm gcn 
        # self.bottom_gcn = GraphConvolution(dim,dim)

        # egcn
        self.bottom_gcn = EGNN(dim = in_dim,edge_dim=1)

        self.act = act
        self.down_gcns = nn.ModuleList()
        self.up_gcns = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.unpools = nn.ModuleList()

        self.subgraph_gcns_1 = nn.ModuleList()
        
        self.norm = nn.LayerNorm(dim)
        self.l_n = len(ks)

        self.down_gcns_1 = nn.ModuleList()
        self.down_gcns_2 = nn.ModuleList()
        self.down_gcns_3 = nn.ModuleList()
        # self.down_gcns_4 = nn.ModuleList()
        # self.down_gcns_5 = nn.ModuleList()
        # self.down_gcns_6 = nn.ModuleList()

        self.up_gcns_1 = nn.ModuleList()
        self.up_gcns_2 = nn.ModuleList()
        self.up_gcns_3 = nn.ModuleList()
        # self.up_gcns_4 = nn.ModuleList()
        # self.up_gcns_5 = nn.ModuleList()
        # self.up_gcns_6 = nn.ModuleList()

        self.bottom_gcns_1 = EGNN(dim = in_dim,edge_dim=1)
        # self.bottom_gcns_2 = EGNN(dim = in_dim,edge_dim=1)
        # self.bottom_gcns_3 = EGNN(dim = in_dim,edge_dim=1)
        # self.bottom_gcns_4 = EGNN(dim = in_dim,edge_dim=1)
        # self.bottom_gcns_5 = EGNN(dim = in_dim,edge_dim=1)
        # self.bottom_gcns_6 = EGNN(dim = in_dim,edge_dim=1)

        # self.egnn_1 = EGNN(dim = in_dim,edge_dim=1)


        for i in range(self.l_n):
            # u_gcn
            # self.down_gcns.append(u_GCN(dim, dim, act, drop_p))
            # self.up_gcns.append(u_GCN(dim, dim, act, drop_p))

            # acm gcn
            # self.down_gcns.append(GraphConvolution(dim,dim))
            # self.up_gcns.append(GraphConvolution(dim,dim))
            # self.subgraph_gcns.append(GraphConvolution(dim,dim))

            # egcn
            self.down_gcns_1.append(EGNN(dim = in_dim,edge_dim=1))
            # self.down_gcns_2.append(EGNN(dim = in_dim,edge_dim=1))
            # self.down_gcns_3.append(EGNN(dim = in_dim,edge_dim=1))
            # self.down_gcns_4.append(EGNN(dim = in_dim,edge_dim=1))
            # self.down_gcns_5.append(EGNN(dim = in_dim,edge_dim=1))
            # self.down_gcns_6.append(EGNN(dim = in_dim,edge_dim=1))
            self.up_gcns_1.append(EGNN(dim = in_dim,edge_dim=1))
            self.pools.append(Pool(ks[i], dim, drop_p))
            self.unpools.append(Unpool(dim, dim, drop_p))

    def readout(self, hs):
        h_max = torch.Tensor([torch.max(h, 0)[0] for h in hs])
        # h_sum = [torch.sum(h, 0) for h in hs]
        # h_mean = [torch.mean(h, 0) for h in hs]
        # h = torch.cat(h_max + h_sum + h_mean)
        return h_max


    # egnn unet 模型
    def forward(self, feat, coor, edge , ep): 

        adj_ms = []
        coor_ms = []
        indices_list = []
        down_outs = []
        hs = []
        org_h = feat
        org_c = coor
        org_e = edge

        for i in range(self.l_n):
            h = feat + self.down_gcns_1[i](feat, coor, edge,ep)
            h = torch.relu(h)
            # h = self.down_gcns_2[i](h, coor, edge,ep)
            # h = torch.relu(h)
            # h = self.down_gcns_3[i](h, coor, edge,ep)
            # h = torch.relu(h)
            # h = self.down_gcns_4[i](h, coor, edge,ep)
            # h = torch.sigmoid(h)
            # h = self.down_gcns_5[i](h, coor, edge,ep)
            # h = torch.sigmoid(h)
            

            g = edge[0,:,:,0]
            h = h[0,:,:]
            coor_ms.append(coor)
            c = coor[0,:,:]
            adj_ms.append(g)
            down_outs.append(h)

            # print(g.shape)
            # print(h)
            # exit()

            g, h, idx = self.pools[i](g, h, ep)
            # print(g.shape)
            # print(h)
            # print(idx)

            feat = h.unsqueeze(0)
            edge = g.unsqueeze(0)
            edge = edge.unsqueeze(edge.dim())
            # print(feat.shape)
            # print(edge.shape)
            # print(c)
            c = c[idx,:]
            coor = c.unsqueeze(0)
            indices_list.append(idx)
        h = feat + self.bottom_gcns_1(feat, coor, edge, ep)
        h = torch.relu(h)
        # h = self.bottom_gcns_2(h, coor, edge, ep)
        # h = torch.relu(h)
        # h = self.bottom_gcns_3(h, coor, edge, ep)
        # h = torch.relu(h)
        # h = self.bottom_gcns_4(h, coor, edge, ep)
        # h = torch.sigmoid(h)
        # h = self.bottom_gcns_5(h, coor, edge, ep)
        # h = torch.sigmoid(h)

        h = h[0,:,:]
        for i in range(self.l_n):
            up_idx = self.l_n - i - 1
            g, idx = adj_ms[up_idx], indices_list[up_idx]
            g, h = self.unpools[i](g, h, down_outs[up_idx], idx)
            # print(h)
            # print(g)
            # exit()
            feat = h.unsqueeze(0)
            edge = g.unsqueeze(0)
            edge = edge.unsqueeze(edge.dim())
            coor = coor_ms[up_idx]

            h = feat
            h = h + self.up_gcns_1[i](feat, coor, edge, ep)
            h = torch.relu(h)
            # h = self.up_gcns_2[i](h, coor, edge, ep)
            # h = torch.relu(h)
            # h = self.up_gcns_3[i](h, coor, edge, ep)
            # h = torch.relu(h)
            # h = self.up_gcns_4[i](h, coor, edge, ep)
            # h = torch.sigmoid(h)
            # h = self.up_gcns_5[i](h, coor, edge, ep)
            # h = torch.sigmoid(h)

            # print(h.shape)
            h = h[0,:,:]
            h = h.add(down_outs[up_idx])
            # h =torch.max(h,down_outs[up_idx])
            # hs.append(h)
        # h = h.add(org_h[0,:,:])

        return hs[1]
        # h = h.unsqueeze(0)
        # h = self.egnn_1(h, org_c, org_e, ep)
        # h = torch.relu(h)
        # h = self.egnn_2(h, org_c, org_e, ep)
        # h = torch.relu(h)
        # h = self.egnn_3(h, org_c, org_e, ep)
        # h = torch.relu(h)
        
        
        return h

        hs.append(h)
        return hs

class u_GCN(nn.Module):

    def __init__(self, in_dim, out_dim, act, p):
        super(u_GCN, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.act = act
        self.drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()

    def forward(self, g, h,ep):

        h = self.drop(h)
        h = torch.matmul(g, h)
        # print(g.size())
        # print(h.size())

        # with torch.no_grad():
        h = self.proj(h)
        h = self.act(h)
        return h



# class Pool(nn.Module):

#     def __init__(self, k, in_dim, p=0.3):
#         super(Pool, self).__init__()
#         self.k = k
#         self.sigmoid = nn.Sigmoid()
#         self.proj = nn.Linear(in_dim, 1)
#         self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

#     def forward(self, g, h, ep):
#         Z = self.drop(h)
#         weights = self.proj(Z).squeeze()
#         # print(123)
#         # print(self.proj.weight.grad)
#         scores = self.sigmoid(weights)
        
#         return top_k_graph(scores, g, h, self.k,ep)


class Pool(nn.Module):

    def __init__(self, k, in_dim, p=0.3):
        super(Pool, self).__init__()
        self.k = k
        self.max_heads = 4
        self.heads = max([h for h in range(1, self.max_heads) if in_dim % h == 0])
        assert self.heads > 0, f"No valid heads for in_dim {in_dim}"
        
        # 多头注意力投影
        self.qkv = nn.Linear(in_dim, in_dim * 3)  # 合并QKV投影
        self.dropout = nn.Dropout(p=p)
        
        # 度数先验的权重（可选）
        self.degree_weight = nn.Parameter(torch.tensor(0.1))
        self.score_proj = nn.Linear(in_dim, 1)

    def forward(self, g, h, ep):
        n_nodes = h.size(0)
        
        # 1. 多头注意力分数
        qkv = self.qkv(h).chunk(3, dim=-1)  # [Q, K, V]
        q, k, v = [x.view(n_nodes, self.heads, -1) for x in qkv]  # [n, h, d_h]
        
        attn = (q @ k.transpose(-2, -1)) / (q.size(-1) ** 0.5)  # [n, h, n]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # 3. 注意力加权聚合（使用V！）
        out = attn @ v  # [n, h, d_h]
        out = out.transpose(1, 2).reshape(n_nodes, -1)  # [n, d]
        
        # 4. 计算最终评分（含度数先验）
        scores = self.score_proj(out).squeeze(-1)  # [n]

        # if hasattr(self, 'degree_weight'):
        #     degree = g.sum(dim=1) if g.dim() == 2 else g[..., 0].sum(dim=1)
        #     scores = scores + self.degree_weight * torch.log(degree + 1e-6)
        
        # 4. 确保输出与原始接口一致
        return top_k_graph(scores, g, h, self.k, ep)



class Unpool(nn.Module):

    def __init__(self, *args):
        super(Unpool, self).__init__()

    def forward(self, g, h, pre_h, idx):
        # print(h[5])
        new_h = h.new_zeros([g.shape[0], h.shape[1]])
        # print(new_h.shape)
        # print(new_h[0:15])
        # print(idx)
        
        new_h[idx] = h
        # print(new_h[idx[5]])
        # print(new_h[85])
        # exit()
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
    # if degrees == 0:
    #     print(g)
    #     print('degrees is zero')
    #     exit()
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

class GraphConvolution(Module):
    def __init__(
        self,
        in_features,
        out_features,
        nnodes= 0,
        model_type='acmgcn',
        output_layer=0,
        variant=False,
        structure_info=0,
    ):
        super(GraphConvolution, self).__init__()
        (
            self.in_features,
            self.out_features,
            self.output_layer,
            self.model_type,
            self.structure_info,
            self.variant,
        ) = (
            in_features,
            out_features,
            output_layer,
            model_type,
            structure_info,
            variant,
        )
        self.count = 0
        
        self.low_vectors = np.empty((0, 2))
        self.high_vectors = np.empty((0, 2))   
        self.mlp_vectors = np.empty((0, 2))
        self.prelu = nn.PReLU()

        self.att_low, self.att_high, self.att_mlp = 0, 0, 0
        self.weight_low, self.weight_high, self.weight_mlp = (
            Parameter(torch.FloatTensor(in_features, out_features).to(device)),
            Parameter(torch.FloatTensor(in_features, out_features).to(device)),
            Parameter(torch.FloatTensor(in_features, out_features).to(device)),
        )
        self.att_vec_low, self.att_vec_high, self.att_vec_mlp = (
            Parameter(torch.FloatTensor(1 * out_features, 1).to(device)),
            Parameter(torch.FloatTensor(1 * out_features, 1).to(device)),
            Parameter(torch.FloatTensor(1 * out_features, 1).to(device)),
            # Parameter(torch.rand(1 * out_features, 1).to(device)),
            # Parameter(torch.rand(1 * out_features, 1).to(device)),
            # Parameter(torch.rand(1 * out_features, 1).to(device)),
        )
        self.layer_norm_low, self.layer_norm_high, self.layer_norm_mlp = (
            nn.LayerNorm(out_features),
            nn.LayerNorm(out_features),
            nn.LayerNorm(out_features),
        )
        self.layer_norm_struc_low, self.layer_norm_struc_high = nn.LayerNorm(
            out_features
        ), nn.LayerNorm(out_features)
        self.att_struc_low = Parameter(
            torch.FloatTensor(1 * out_features, 1).to(device)
        )
        self.struc_low = Parameter(torch.FloatTensor(nnodes, out_features).to(device))
        if self.structure_info == 0:
            self.att_vec = Parameter(torch.FloatTensor(3, 3).to(device))
            # self.att_vec = Parameter(torch.rand(3, 3).to(device))
        else:
            self.att_vec = Parameter(torch.FloatTensor(4, 4).to(device))
        self.reset_parameters()

    def reset_parameters(self):

        stdv = 1.0 / math.sqrt(self.weight_mlp.size(1))
        std_att = 1.0 / math.sqrt(self.att_vec_mlp.size(1))
        std_att_vec = 1.0 / math.sqrt(self.att_vec.size(1))

        self.weight_low.data.uniform_(-stdv, stdv)
        self.weight_high.data.uniform_(-stdv, stdv)
        self.weight_mlp.data.uniform_(-stdv, stdv)
        self.struc_low.data.uniform_(-stdv, stdv)

        self.att_vec_high.data.uniform_(-std_att, std_att)
        self.att_vec_low.data.uniform_(-std_att, std_att)
        self.att_vec_mlp.data.uniform_(-std_att, std_att)
        self.att_struc_low.data.uniform_(-std_att, std_att)

        # print(self.att_vec)
        self.att_vec.data.uniform_(-std_att_vec, std_att_vec)

        # print(self.att_vec)
        # print(self.att_vec_low.size())
        # print(self.att_vec_high.size())
        # print(self.att_vec_mlp.size())
        # exit()

        self.layer_norm_low.reset_parameters()
        self.layer_norm_high.reset_parameters()
        self.layer_norm_mlp.reset_parameters()
        self.layer_norm_struc_low.reset_parameters()
        self.layer_norm_struc_high.reset_parameters()

    def attention3(self, output_low, output_high, output_mlp, ep):
        T = 3
        if self.model_type == "acmgcn+" or self.model_type == "acmgcn++":
            output_low, output_high, output_mlp = (
                self.layer_norm_low(output_low),
                self.layer_norm_high(output_high),
                self.layer_norm_mlp(output_mlp),
            )
        
        # print(torch.sigmoid(torch.mm((output_low), self.att_vec_low)))
        # print(self.att_vec)
        # exit()
        # torch.sigmoid()
        logits = (
            torch.mm(
                torch.sigmoid(
                    torch.cat(
                        [
                            torch.mm((output_low), self.att_vec_low),
                            torch.mm((output_high), self.att_vec_high),
                            torch.mm((output_mlp), self.att_vec_mlp),
                        ],
                        1,
                    )
                ),
                self.att_vec,
            )
            / T
        )

        att = torch.softmax(logits, 1)

        return att[:, 0][:, None], att[:, 1][:, None], att[:, 2][:, None]

    def attention4(self, output_low, output_high, output_mlp, struc_low):
        T = 4
        if self.model_type == "acmgcn+" or self.model_type == "acmgcn++":
            feature_concat = torch.cat(
                [
                    torch.mm(self.layer_norm_low(output_low), self.att_vec_low),
                    torch.mm(self.layer_norm_high(output_high), self.att_vec_high),
                    torch.mm(self.layer_norm_mlp(output_mlp), self.att_vec_mlp),
                    torch.mm(self.layer_norm_struc_low(struc_low), self.att_struc_low),
                ],
                1,
            )
        else:
            feature_concat = torch.cat(
                [
                    torch.mm((output_low), self.att_vec_low),
                    torch.mm((output_high), self.att_vec_high),
                    torch.mm((output_mlp), self.att_vec_mlp),
                    torch.mm((struc_low), self.att_struc_low),
                ],
                1,
            )

        logits = torch.mm(torch.sigmoid(feature_concat), self.att_vec) / T

        att = torch.softmax(logits, 1)
        return (
            att[:, 0][:, None],
            att[:, 1][:, None],
            att[:, 2][:, None],
            att[:, 3][:, None],
        )

    def forward(self, g, feats,ep):
        output = 0
        self.count = self.count + 1

        nnodes = feats.shape[0] 

        x = normalize_tensor(feats[:,:]).to(device)
        
        adj_low_unnormalized = g

        adj_low = normalize_tensor(torch.eye(nnodes).to(device) + adj_low_unnormalized)
        adj_high = (torch.eye(nnodes).to(device)  - adj_low).to(device).to_sparse()
        adj_low = adj_low.to(device)
        adj_low_unnormalized = adj_low_unnormalized.to(device)

        output_low = torch.relu(
                torch.spmm(adj_low, (torch.mm(x, self.weight_low)))
                )
        output_high = torch.relu(
                torch.spmm(adj_high, (torch.mm(x, self.weight_high)))
                )
        output_mlp = torch.relu(torch.mm(x, self.weight_mlp))


        self.att_low, self.att_high, self.att_mlp = self.attention3(
                (output_low), (output_high), (output_mlp),ep
            )


        return 3 * (
            self.att_low * output_low
            + self.att_high * output_high
            + self.att_mlp * output_mlp
        )

       

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class MLP(nn.Module):
    """adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py"""

    def __init__(
        self, in_channels, hidden_channels, out_channels, num_layers, dropout=0.5
    ):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
            self.bns.append(nn.BatchNorm1d(out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data, input_tensor=False):
        if not input_tensor:
            x = data.graph["node_feat"]
        else:
            x = data
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x