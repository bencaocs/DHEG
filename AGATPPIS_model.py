import pickle
import math

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter
from graph_unet import u_GCN, GraphUnet, Subgraphnet,Initializer
from graph_unet import *
from EGNN import *
import warnings
warnings.filterwarnings("ignore")
import dgl


Feature_Path = "./Feature/"
SEED = 2020 # 27 # 2020 #2021
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(SEED)

BASE_MODEL_TYPE = 'AGAT'  # agat/gcn
ADD_NODEFEATS = 'all'  # all/atom_feats/psepose_embedding/no/EMSC/EMSCA64noRMSN/newall
USE_EFEATS = True  # True/False
    
MAP_CUTOFF = 14 # 14
DIST_NORM = 15

# INPUT_DIM
if ADD_NODEFEATS == 'all':  # add atom features and psepose embedding
    INPUT_DIM = 54 +7 #+2 + 7 
elif ADD_NODEFEATS == 'atom_feats':  # only add atom features
    INPUT_DIM = 54 + 7
elif ADD_NODEFEATS == 'psepose_embedding':  # only add psepose embedding
    INPUT_DIM = 54 + 1
elif ADD_NODEFEATS == 'no':
    INPUT_DIM = 54
elif ADD_NODEFEATS == 'newall':
    INPUT_DIM = 78 #32 + 14 + 20 +7+1 #32 + 14 + 20 +7+1  #32 + 20 + 7 + 1
elif ADD_NODEFEATS == 'comattention':
    INPUT_DIM = 62
elif ADD_NODEFEATS == 'seqstr':
    INPUT_DIM = 32 + 14 + 7 + 1


HIDDEN_DIM = 256  # hidden size of node features
LAYER = 8  # the number of AGAT layers
DROPOUT = 0.1
ALPHA = 0.7
LAMBDA = 1.5

LEARNING_RATE = 5E-4 #4 #-3 5E-5 5E-4
WEIGHT_DECAY = 1E-4 #1E-4 #0 
BATCH_SIZE = 1
NUM_CLASSES = 2  # [not bind, bind]
NUMBER_EPOCHS = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Feature_Path='/home/bio-3090ti/BenzcodeL/MEGPPIS/Feature/'
def embedding(sequence_name):
    pssm_feature = np.load(Feature_Path + "pssm/" + sequence_name + '.npy')
    hmm_feature = np.load(Feature_Path + "hmm/" + sequence_name + '.npy')

    seq_embedding = np.concatenate([pssm_feature, hmm_feature], axis=1)

    return seq_embedding.astype(np.float32)

def get_EMSC_feature(sequence_name):
    PreESM_embedding = np.load(Feature_Path + "EMSC63noRMSN/" + sequence_name + '.npy')  #EMSCA64[768-256-64]noRMSN   EMSC62noRMSN
    #PreESM_embedding = np.load(Feature_Path + "EMS263noRMSN/" + sequence_name + '.npy') 
    
    # PreESM_embedding = PreESM_embedding[1:-1,:]
    # seq_embedding = np.concatenate([hmm_feature, PreESM_embedding], axis=1)

    return PreESM_embedding.astype(np.float32)


def get_dssp_features(sequence_name):
    dssp_feature = np.load(Feature_Path + "dssp/" + sequence_name + '.npy')
    return dssp_feature.astype(np.float32)

def get_res_atom_features(sequence_name):
    res_atom_feature = np.load(Feature_Path + "resAF/" + sequence_name + '.npy')
    return res_atom_feature.astype(np.float32)

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = np.diag(r_inv)
    result = r_mat_inv @ mx @ r_mat_inv
    return result


def cal_edges(sequence_name, radius=MAP_CUTOFF):  # to get the index of the edges
    dist_matrix = np.load(Feature_Path + "distance_map_SC/" + sequence_name + ".npy")
    mask = ((dist_matrix >= 0) * (dist_matrix <= radius))
    adjacency_matrix = mask.astype(int)

    radius_index_list = np.where(adjacency_matrix == 1)
    radius_index_list = [list(nodes) for nodes in radius_index_list]
    return radius_index_list

def load_graph(sequence_name):
    dismap = np.load(Feature_Path + "distance_map_SC/" + sequence_name + ".npy")

    mask = ((dismap >= 0) * (dismap <= MAP_CUTOFF))
    adjacency_matrix = mask.astype(int)# mask.astype(np.int)
    norm_matrix = normalize(adjacency_matrix.astype(np.float32))
    return norm_matrix

import torch
from torch.nn.utils.rnn import pad_sequence
import dgl

def graph_collate(samples):
    sequence_name, sequence, label, node_features, G, adj_matrix , pos = map(list, zip(*samples))
    label = torch.Tensor(label)
    G_batch = dgl.batch(G)
    node_features = torch.cat(node_features)
    adj_matrix = torch.Tensor(adj_matrix)
    pos = torch.cat(pos)
    pos = torch.Tensor(pos)
    return sequence_name, sequence, label, node_features, G_batch, adj_matrix, pos

class ProDataset(Dataset):
    def __init__(self, dataframe, radius=MAP_CUTOFF, dist=DIST_NORM, psepos_path='/home/bio-3090ti/BenzcodeL/MEGPPIS/Feature/psepos/Train335_psepos_SC.pkl'):
        self.names = dataframe['ID'].values
        self.sequences = dataframe['sequence'].values
        self.labels = dataframe['label'].values
        # psepos_path='./Feature/psepos/Train335_psepos_CA.pkl'
        # psepos_path='./Feature/psepos/Train335_psepos_C.pkl'
        self.residue_psepos = pickle.load(open(psepos_path, 'rb'))
        self.radius = radius
        self.dist = dist

    def __getitem__(self, index):
        sequence_name = self.names[index]
        sequence = self.sequences[index]
        # label = torch.tensor(self.labels[index], dtype=torch.long)#Ben-0818
        label = np.array(self.labels[index]) 
        nodes_num = len(sequence)
        # print(self.residue_psepos)
        pos = self.residue_psepos[sequence_name]
        # print(pos)
        
        # agat
        reference_res_psepos = pos[0]
        pos1 = pos - reference_res_psepos
        pos1 = torch.from_numpy(pos1)
        
        pos = torch.from_numpy(pos).type(torch.FloatTensor)

        sequence_embedding = embedding(sequence_name)

        ########
        sequence_embedding = embedding(sequence_name)
        structural_features = get_dssp_features(sequence_name)
        # +dssp 
        node_features = np.concatenate([sequence_embedding, structural_features], axis=1)
        
        node_features = torch.from_numpy(node_features)
        if ADD_NODEFEATS == 'all' or ADD_NODEFEATS == 'atom_feats' or ADD_NODEFEATS == 'seqstr' or ADD_NODEFEATS =='comattention':
            res_atom_features = get_res_atom_features(sequence_name)
            res_atom_features = torch.from_numpy(res_atom_features)
            node_features = torch.cat([node_features, res_atom_features], dim=-1)
        if ADD_NODEFEATS == 'all' or ADD_NODEFEATS == 'psepose_embedding' or ADD_NODEFEATS == 'seqstr' or ADD_NODEFEATS =='comattention':
            dist_abs = torch.sqrt(torch.sum(pos * pos, dim=1)).unsqueeze(-1) / self.dist  # 绝对距离
            dist_rel = torch.sqrt(torch.sum(pos1 * pos1, dim=1)).unsqueeze(-1) / self.dist  # 相对距离

            # node_features = torch.cat([node_features, torch.sqrt(
            #     torch.sum(pos * pos, dim=1)).unsqueeze(-1) / self.dist], dim=-1)

            node_features = torch.cat([node_features, dist_abs, dist_rel], dim=-1) #, dist_rel

        radius_index_list = cal_edges(sequence_name, MAP_CUTOFF)
        edge_feat = self.cal_edge_attr(radius_index_list, pos)

        G = dgl.DGLGraph()
        G.add_nodes(nodes_num)
        edge_feat = np.transpose(edge_feat, (1, 2, 0))
        edge_feat = edge_feat.squeeze(1)

        self.add_edges_custom(G,
                              radius_index_list,
                              edge_feat
                              )

        adj_matrix = load_graph(sequence_name)


        node_features = node_features.detach().numpy()

        node_features = node_features[np.newaxis, :, :]
        node_features = torch.from_numpy(node_features).type(torch.FloatTensor)



        return sequence_name, sequence, label, node_features, G, adj_matrix , pos

    def __len__(self):
        return len(self.labels)

    def cal_edge_attr(self, index_list, pos):
        pdist = nn.PairwiseDistance(p=2,keepdim=True)
        cossim = nn.CosineSimilarity(dim=1)

        distance = (pdist(pos[index_list[0]], pos[index_list[1]]) / self.radius).detach().numpy()
        cos = ((cossim(pos[index_list[0]], pos[index_list[1]]).unsqueeze(-1) + 1) / 2).detach().numpy()
        radius_attr_list = np.array([distance, cos])
        return radius_attr_list

    def add_edges_custom(self, G, radius_index_list, edge_features):
        src, dst = radius_index_list[1], radius_index_list[0]
        if len(src) != len(dst):
            print('source and destination array should have been of the same length: src and dst:', len(src), len(dst))
            raise Exception
        G.add_edges(src, dst)
        G.edata['ex'] = torch.tensor(edge_features)


class GNet(nn.Module):
    def __init__(self, in_dim, n_classes, args = {'drop_n':0.0 ,'ks': [1,0.6], 'l_dim':63,'h_dim':63}): #[1,0.5] 63
        super(GNet, self).__init__()
        self.n_act = torch.relu
        self.c_act = torch.relu
        # self.s_gcn = u_GCN(in_dim, args['l_dim'], self.n_act, 0.0)
        self.g_unet = Subgraphnet(
            args['ks'], args['l_dim'], args['l_dim'], args['l_dim'], self.n_act,
            0.1) #0.1
######################################################################################################################### 
        self.out_l_1 = nn.Linear(args['l_dim'], 20) #20
        self.out_l_2 = nn.Linear(20, 10) #20
        self.out_l_3 = nn.Linear(10, n_classes)

        self.out_drop = nn.Dropout(p=0.0)

    # egnn unet
    def forward(self, residue_features, coordinate_features, adjacency_matrix,ep):
        
        # egnn处理
        hs = self.g_unet(residue_features, coordinate_features, adjacency_matrix, ep)
        hs = self.out_l_1(hs)
        hs = self.c_act(hs)
        hs = self.out_l_2(hs)
        hs = self.c_act(hs)
        hs = self.out_l_3(hs)
        return hs


class AGATPPIS(nn.Module):
    def __init__(self, nlayers, nfeat, nhidden, nclass, dropout, lamda, alpha):
        super(AGATPPIS, self).__init__()

        self.input_dim = nfeat
        self.output_dim = nclass
        current_dim = nfeat
        
        self.subegnn = GNet(nfeat,nclass)

        self.graph_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        
        graph_hidden_layer_output_dims = [nfeat]*6
        linear_hidden_layer_output_dims = [10]*2
        
            
        # dense layers
        for hdim in linear_hidden_layer_output_dims:
            self.linear_layers.append(nn.Linear(in_features = current_dim,
                                                out_features = hdim))
            current_dim = hdim

        self.linear_layers.append(nn.Linear(in_features = current_dim, out_features = nclass))
        
        self.criterion = nn.CrossEntropyLoss()  # automatically do softmax to the predicted value and one-hot to the label
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.6, patience=5, min_lr=1e-6)


    def forward(self, x, graph, adj_matrix, pos,mask=None):
        
        feats = x
        edges = adj_matrix.unsqueeze(0)
        edges = edges.unsqueeze(edges.dim())
        coors = pos.unsqueeze(0)

        # return self.gunet(feats, coors, edges, 0)
        return self.subegnn(feats, coors, edges, 0)
