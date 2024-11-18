import torch
from torch.utils.data import Dataset
import torch.utils.data as D
from torch import nn
import pandas as pd
import numpy as np
import copy
import os
from sklearn.metrics import roc_auc_score, log_loss
from tqdm import tqdm
import sys
import random
import warnings

warnings.filterwarnings('ignore')
import sys
import time
from itertools import combinations

torch.multiprocessing.set_sharing_strategy('file_system')
from DynamicConv import DynamicConv
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, CompressedInteractionNet, LogisticRegression
import torch.nn.functional as F
import math
from torch.cuda.amp import autocast, GradScaler
from fftKAN import NaiveFourierKANLayer
file_path = __file__
file_name = file_path.split('/')[-1]


# def seed_everything(seed):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
# seed_everything(1)

PATH = sys.path[0]

# def worker_init_fn(worker_id):
#     worker_seed = torch.initial_seed() % 2**32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)

config = {

    'data_path': '/data/pro1/dataset/criteo/criteo.pkl',
    'sparse_cols': [f'C{x}' for x in range(1, 27)],
    'dense_cols': [f'I{x}' for x in range(1, 14)],
    'train_ratio': 0.8,
    'valid_ratio': 0.1,
    'debug_mode': True,
    'epoch': 100,
    # 'batch': 32768,
    'batch': 4096,
    'lr': 0.001,
    'device': 0,
    'num_workers': 8,
    'embedding_dim':16
}

df = pd.read_pickle(config['data_path'])

if config['debug_mode']:
    df = df[:1000]
df[config['sparse_cols']] = df[config['sparse_cols']].fillna('-1', )
df[config['dense_cols']] = df[config['dense_cols']].fillna(0, )


train_num = int(len(df) * config['train_ratio'])
valid_num = int(len(df) * config['valid_ratio'])


train_df = df[:train_num].reset_index(drop=True)

valid_df = df[train_num:train_num + valid_num].reset_index(drop=True)
test_df = df[train_num + valid_num:].reset_index(drop=True)






# Dataset构造
class BaseDataset(Dataset):
    def __init__(self, config, df, enc_dict=None):
        self.config = config
        self.df = df
        self.enc_dict = enc_dict
        self.dense_cols = list(set(self.config['dense_cols']))
        self.sparse_cols = list(set(self.config['sparse_cols']))
        self.feature_name = self.dense_cols + self.sparse_cols + ['label']

        # 数据编码
        if self.enc_dict == None:
            self.get_enc_dict()
        self.enc_data()

    def get_enc_dict(self):
        # 计算enc_dict
        self.enc_dict = dict(zip(list(self.dense_cols + self.sparse_cols),
                                 [dict() for _ in range(len(self.dense_cols + self.sparse_cols))]))
        for f in self.sparse_cols:
            self.df[f] = self.df[f].astype('str')
            map_dict = dict(zip(self.df[f].unique(), range(1, self.df[f].nunique() + 1)))
            self.enc_dict[f] = map_dict
            self.enc_dict[f]['vocab_size'] = self.df[f].nunique() + 1
            '''
            eg:C17 特征的map_dict
            {'e5ba7672': 1,
             '07c540c4': 2,
             '8efede7f': 3,
             '1e88c74f': 4,
             '776ce399': 5,
             'd4bb7bd8': 6,
             '3486227d': 7,
             '27c07bd6': 8,
             '2005abd1': 9,
             'vocab_size': 10}
            '''

        for f in self.dense_cols:
            self.enc_dict[f]['min'] = self.df[f].min()
            self.enc_dict[f]['max'] = self.df[f].max()
            '''
            eg:I6 特征
            {'min': 0.0, 'max': 4638.0}
            '''
        return self.enc_dict

    def enc_dense_data(self, col):
        return (self.df[col] - self.enc_dict[col]['min']) / (self.enc_dict[col]['max'] - self.enc_dict[col]['min'])

    def enc_sparse_data(self, col):
        return self.df[col].apply(lambda x: self.enc_dict[col].get(x, 0))

    def enc_data(self):
        # 使用enc_dict对数据进行编码
        self.enc_df = copy.deepcopy(self.df)
        for col in self.dense_cols:
            self.enc_df[col] = self.enc_dense_data(col)
        for col in self.sparse_cols:
            self.enc_df[col] = self.enc_sparse_data(col)

    def __getitem__(self, index):
        data = dict()
        for col in self.feature_name:
            if col in self.dense_cols:
                data[col] = torch.Tensor([self.enc_df[col].iloc[index]]).squeeze(-1)
            elif col in self.sparse_cols:
                data[col] = torch.Tensor([self.enc_df[col].iloc[index]]).long().squeeze(-1)
        data['label'] = torch.Tensor([self.enc_df['label'].iloc[index]]).squeeze(-1)
        return data

    def __len__(self):
        return len(self.enc_df)


train_dataset = BaseDataset(config, train_df)
enc_dict = train_dataset.get_enc_dict()

valid_dataset = BaseDataset(config, valid_df, enc_dict=enc_dict)
test_dataset = BaseDataset(config, test_df, enc_dict=enc_dict)


# 基本网络模块

# 通用Emb
class EmbeddingLayer(nn.Module):
    def __init__(self,
                 enc_dict=None,
                 embedding_dim=None):
        super(EmbeddingLayer, self).__init__()
        self.enc_dict = enc_dict
        self.embedding_dim = embedding_dim
        self.embedding_layer = nn.ModuleDict()

        self.emb_feature = []

        for col in self.enc_dict.keys():
            if 'vocab_size' in self.enc_dict[col].keys():
                self.emb_feature.append(col)
                self.embedding_layer.update({col: nn.Embedding(
                    self.enc_dict[col]['vocab_size'],
                    self.embedding_dim,
                )})

    @autocast()
    def forward(self, X):
        # 对所有的sparse特征挨个进行embedding
        feature_emb_list = []
        for col in self.emb_feature:
            inp = X[col].long().view(-1, 1)
            feature_emb_list.append(self.embedding_layer[col](inp))
        return feature_emb_list


class EmbeddingLayerV2(nn.Module):
    def __init__(self,
                 enc_dict=None,
                 embedding_dim=None):
        super(EmbeddingLayerV2, self).__init__()
        self.enc_dict = enc_dict
        self.embedding_dim = embedding_dim
        self.embedding_layer = nn.ModuleDict()

        self.emb_feature = []

        for col in self.enc_dict.keys():
            if 'vocab_size' in self.enc_dict[col].keys():
                self.emb_feature.append(col)
                self.embedding_layer.update({col: nn.Embedding(
                    self.enc_dict[col]['vocab_size'],
                    self.embedding_dim,
                )})
            else:
                self.emb_feature.append(col)
                self.embedding_layer.update({col: nn.Linear(1, self.embedding_dim)
                                             })

    @autocast()
    def forward(self, X):
        # 对所有的sparse特征挨个进行embedding
        feature_emb_list = []
        for col in self.emb_feature:
            inp = X[col].view(-1, 1)
            feature_emb_list.append(self.embedding_layer[col](inp).squeeze(1))
        return feature_emb_list


class Gate(nn.Module):
    def __init__(self, embedding_dim, net_dropout, batch_norm,input_dim, enc_dict):
        super(Gate, self).__init__()
        self.input_dim = input_dim
        self.feature_num = len(enc_dict)
        self.easy_embedding_dim = int(embedding_dim/2)
        self.hard_embedding_dim = embedding_dim
        self.gate = MLP_Block(input_dim=input_dim,
                              output_dim=1,
                              hidden_units=[input_dim // 4],
                              hidden_activations="ReLU",
                              output_activation="sigmoid",
                              dropout_rates=net_dropout,
                              batch_norm=batch_norm)
        self.easy_mlp = MLP_Block(input_dim=input_dim,
                              output_dim=self.feature_num*self.easy_embedding_dim,
                              hidden_units=[input_dim // 4],
                              hidden_activations="ReLU",
                              output_activation="sigmoid",
                              dropout_rates=net_dropout,
                              batch_norm=batch_norm)
        self.hard_mlp = MLP_Block(input_dim=input_dim,
                                 output_dim=self.feature_num*self.hard_embedding_dim,
                                 hidden_units=[input_dim // 4],
                                 hidden_activations="ReLU",
                                 output_activation="sigmoid",
                                 dropout_rates=net_dropout,
                                 batch_norm=batch_norm)

    @autocast()
    def forward(self, X):
        gate = self.gate(X)
        easy_out = X * gate
        hard_out = X * (1 - gate)
        easy_out = self.easy_mlp(easy_out).view(X.shape[0], self.feature_num, self.easy_embedding_dim)
        hard_out = self.hard_mlp(hard_out).view(X.shape[0], self.feature_num, self.hard_embedding_dim)
        return easy_out, hard_out




class WSF(nn.Module):
    def __init__(self, sim_encoder_dim, com_encoder_dim):
        super(WSF, self).__init__()
        self.fc1 = nn.Linear(sim_encoder_dim, 1)
        self.fc2 = nn.Linear(com_encoder_dim, 1)
        self.w = nn.Parameter(torch.empty(2, 1).fill_(0.5), requires_grad=True)

    @autocast()
    def forward(self, sim_encoder_out, com_encoder_out):
        y_pred_1 = self.fc1(sim_encoder_out)
        y_pred_2 = self.fc2(com_encoder_out)
        y_pred = torch.matmul(torch.cat([y_pred_1, y_pred_2], dim=-1), self.w)
        return y_pred, y_pred_1, y_pred_2

class sum_fusion(nn.Module):
    def __init__(self, sim_encoder_dim, com_encoder_dim):
        super(sum_fusion, self).__init__()
        self.fc1 = nn.Linear(sim_encoder_dim, 1)
        self.fc2 = nn.Linear(com_encoder_dim, 1)

    @autocast()
    def forward(self, sim_encoder_out, com_encoder_out):
        y_pred_1 = self.fc1(sim_encoder_out)
        y_pred_2 = self.fc2(com_encoder_out)
        return y_pred_1 + y_pred_2, y_pred_1, y_pred_2


# 一阶交叉
class LR_Layer(nn.Module):
    def __init__(self, enc_dict):
        super(LR_Layer, self).__init__()
        self.enc_dict = enc_dict
        self.emb_layer = EmbeddingLayer(enc_dict=self.enc_dict, embedding_dim=1)
        self.dnn_input_dim = get_dnn_input_dim(self.enc_dict, 1)
        self.fc = nn.Linear(self.dnn_input_dim, 1)

    @autocast()
    def forward(self, data):
        sparse_emb = self.emb_layer(data)
        sparse_emb = torch.stack(sparse_emb, dim=1).flatten(1)  # [batch,num_sparse*emb]
        dense_input = get_linear_input(self.enc_dict, data)  # [batch,num_dense]
        dnn_input = torch.cat((sparse_emb, dense_input), dim=1)  # [batch,num_sparse*emb + num_dense]
        out = self.fc(dnn_input)
        return out


# CIN
class CIN(nn.Module):
    def __init__(self,
                 sparse_num,
                 dense_num,
                 cin_hidden_units=[16, 16, 16],  # H1,H2,...,HK
                 output_dim = 1
                 ):
        super(CIN, self).__init__()
        self.sparse_num = sparse_num
        self.dense_num = dense_num
        self.cin_hidden_units = cin_hidden_units
        self.fc = nn.Linear(sum(self.cin_hidden_units), output_dim)

        self.cin_layer = nn.ModuleList()
        for i, unit in enumerate(self.cin_hidden_units):
            in_channels = (self.sparse_num + self.dense_num) * self.cin_hidden_units[i - 1] if i > 0 else (
                                                                                                                      self.sparse_num + self.dense_num) ** 2
            out_channels = unit
            self.cin_layer.append(nn.Conv1d(in_channels,
                                            out_channels,
                                            kernel_size=1))

    @autocast()
    def forward(self, sparse_embedding):
        batch_size = sparse_embedding.shape[0]
        embedding_dim = sparse_embedding.shape[-1]

        cin_output_list = []

        X0 = sparse_embedding  # batch,m,emb
        Xk = X0  # batch,Hk,emb
        for idx, cin in enumerate(self.cin_layer):
            Zk = torch.einsum('bhce,bcde->bhde', Xk.unsqueeze(-2), X0.unsqueeze(1))  # batch,Hk,m,emb
            Zk = Zk.view(batch_size, -1, embedding_dim)  # batch,Hk*m,emb
            Xk = self.cin_layer[idx](Zk)  # batch,Hk+1,emb

            cin_output_list.append(Xk.sum(dim=-1))  # batch,Hk+1

        cin_output = torch.cat(cin_output_list, dim=-1)  # batch,sum(self.cin_hidden_units)
        output = self.fc(cin_output)
        return output

class CIN2(nn.Module):
    def __init__(self,
                 sparse_num,
                 dense_num,
                 embedding_dim,
                 cin_hidden_units=[16, 16, 16],  # H1,H2,...,HK
                 output_dim = 1
                 ):
        super(CIN2, self).__init__()
        self.sparse_num = sparse_num
        self.dense_num = dense_num
        self.Dconv_input_embedding_dim = int(embedding_dim / 2)
        self.cin_hidden_units = cin_hidden_units
        self.fc = nn.Linear(sum(self.cin_hidden_units), output_dim)

        self.cin_layer = nn.ModuleList()
        for i, unit in enumerate(self.cin_hidden_units):
            in_channels = (self.sparse_num + self.dense_num) * self.cin_hidden_units[i - 1] if i > 0 else (self.sparse_num + self.dense_num) ** 2
            out_channels = unit
            # self.cin_layer.append(nn.Conv1d(in_channels,
            #                                 out_channels,
            #                                 kernel_size=1))
            self.cin_layer.append(DynamicConv(in_planes=self.Dconv_input_embedding_dim,out_planes=out_channels,kernel_size=3,stride=1,padding=1,bias=False))

    @autocast()
    def forward(self, sparse_embedding):
        batch_size = sparse_embedding.shape[0]
        embedding_dim = sparse_embedding.shape[-1]

        cin_output_list = []

        X0 = sparse_embedding  # batch,m,emb
        Xk = X0  # batch,Hk,emb
        for idx, cin in enumerate(self.cin_layer):

            Zk = torch.einsum('bhce,bcde->bedh', Xk.unsqueeze(-2), X0.unsqueeze(1))  # batch,Hk,m,emb
            # Zk = Zk.view(batch_size, -1, embedding_dim)  # batch,Hk*m,emb
            Xk = self.cin_layer[idx](Zk)  # batch,Hk+1,emb

            cin_output_list.append(Xk.sum(dim=-1))  # batch,Hk+1

        cin_output = torch.cat(cin_output_list, dim=-1)  # batch,sum(self.cin_hidden_units)
        output = self.fc(cin_output)
        return output


# DNN
class MLP_Layer(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim=None,
                 hidden_units=[],
                 hidden_activations="ReLU",
                 final_activation=None,
                 dropout_rates=0,
                 batch_norm=False,
                 use_bias=True):
        super(MLP_Layer, self).__init__()
        dense_layers = []
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        hidden_activations = [set_activation(x) for x in hidden_activations]
        hidden_units = [input_dim] + hidden_units
        for idx in range(len(hidden_units) - 1):
            dense_layers.append(nn.Linear(hidden_units[idx], hidden_units[idx + 1], bias=use_bias))
            if batch_norm:
                dense_layers.append(nn.BatchNorm1d(hidden_units[idx + 1]))
            if hidden_activations[idx]:
                dense_layers.append(hidden_activations[idx])
            if dropout_rates[idx] > 0:
                dense_layers.append(nn.Dropout(p=dropout_rates[idx]))
        if output_dim is not None:
            dense_layers.append(nn.Linear(hidden_units[-1], output_dim, bias=use_bias))
        if final_activation is not None:
            dense_layers.append(set_activation(final_activation))
        self.dnn = nn.Sequential(*dense_layers)  # * used to unpack list

    @autocast()
    def forward(self, inputs):
        return self.dnn(inputs)
class MLP2_Layer(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim=None,
                 hidden_units=[],
                 hidden_activations="ReLU",
                 final_activation=None,
                 dropout_rates=0,
                 batch_norm=False,
                 use_bias=True):
        super(MLP2_Layer, self).__init__()
        dense_layers = []
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        hidden_activations = [set_activation(x) for x in hidden_activations]
        hidden_units = [input_dim] + hidden_units
        for idx in range(len(hidden_units) - 1):
            dense_layers.append(NaiveFourierKANLayer(hidden_units[idx], hidden_units[idx + 1],gridsize=5))
            if batch_norm:
                dense_layers.append(nn.BatchNorm1d(hidden_units[idx + 1]))
            if hidden_activations[idx]:
                dense_layers.append(hidden_activations[idx])
            if dropout_rates[idx] > 0:
                dense_layers.append(nn.Dropout(p=dropout_rates[idx]))
        if output_dim is not None:
            dense_layers.append(NaiveFourierKANLayer(hidden_units[idx], hidden_units[idx + 1],gridsize=5))
        if final_activation is not None:
            dense_layers.append(set_activation(final_activation))
        self.dnn = nn.Sequential(*dense_layers)  # * used to unpack list

    @autocast()
    def forward(self, inputs):
        return self.dnn(inputs)

class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    @autocast()
    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                NaiveFourierKANLayer(in_features, out_features, gridsize=self.grid_size)
            )

    @autocast()
    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )


def set_device(gpu=-1):
    if gpu >= 0 and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")
    return device


def set_activation(activation):
    if isinstance(activation, str):
        if activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "sigmoid":
            return nn.Sigmoid()
        elif activation.lower() == "tanh":
            return nn.Tanh()
        else:
            return getattr(nn, activation)()
    else:
        return activation


def get_dnn_input_dim(enc_dict, embedding_dim):
    num_sparse = 0
    num_dense = 0
    for col in enc_dict.keys():
        if 'min' in enc_dict[col].keys():
            num_dense += 1
        elif 'vocab_size' in enc_dict[col].keys():
            num_sparse += 1
    return num_sparse * embedding_dim + num_dense * embedding_dim


def get_feature_num(enc_dict):
    num_sparse = 0
    num_dense = 0
    for col in enc_dict.keys():
        if 'min' in enc_dict[col].keys():
            num_dense += 1
        elif 'vocab_size' in enc_dict[col].keys():
            num_sparse += 1
    return num_sparse, num_dense


def get_linear_input(enc_dict, data):
    res_data = []
    for col in enc_dict.keys():
        if 'min' in enc_dict[col].keys():
            res_data.append(data[col])
    res_data = torch.stack(res_data, axis=1)
    return res_data


# xDeepFM模型
class xDeepFM(nn.Module):
    def __init__(self,
                 embedding_dim=10,
                 hidden_units=[64, 64, 64],
                 cin_hidden_units=[8, 8, 8, 8, 8],
                 # loss_fun='torch.nn.BCELoss()',
                 loss_fun='torch.nn.BCEWithLogitsLoss()',
                 enc_dict=None,
                 DFM=None,
                 c=0.8,
                 gamma=2,
                 alpha=0.2,
                 hard_hidden_units=[64, 64, 64],
                 net_dropout=0,
                 batch_norm=False
                 ):
        super(xDeepFM, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.hard_hidden_units=hard_hidden_units
        self.cin_hidden_units = cin_hidden_units
        self.loss_fun = eval(loss_fun)
        self.enc_dict = enc_dict
        self.c = c
        self.gamma = gamma
        self.alpha = alpha
        self.embedding_layer = EmbeddingLayerV2(enc_dict=self.enc_dict, embedding_dim=self.embedding_dim)
        self.output_activation = nn.Sigmoid()
        self.sparse_num, self.dense_num = get_feature_num(self.enc_dict)  # 论文中的“m“
        self.net_dropout = net_dropout
        self.batch_norm = batch_norm

        self.cin = CIN(self.sparse_num, self.dense_num, self.cin_hidden_units)  # CIN
        self.lr = LR_Layer(enc_dict=enc_dict)  # 一阶

        self.dnn_input_dim = get_dnn_input_dim(self.enc_dict, self.embedding_dim)
        # sparse_num * emb_dim + dense_num

        self.dnn = MLP_Layer(input_dim=self.dnn_input_dim, output_dim=1, hidden_units=self.hidden_units,
                             hidden_activations='relu', dropout_rates=self.net_dropout)

        # self.easy_encoder = CompressedInteractionNet(len(enc_dict), cin_hidden_units,output_dim=hard_hidden_units[-1])
        self.easy_encoder = CIN2(self.sparse_num, self.dense_num,self.embedding_dim, self.cin_hidden_units,output_dim=self.hard_hidden_units[-1])  # CIN
        self.hard_encoder = KAN([self.dnn_input_dim, self.hard_hidden_units[-1], self.hard_hidden_units[-1]], grid_size=8)
        self.GM = Gate(enc_dict=self.enc_dict, embedding_dim=self.embedding_dim,
                       net_dropout=self.net_dropout,
                       batch_norm=self.batch_norm,
                       input_dim=self.dnn_input_dim)



        if DFM == 'WSF':
            self.DFM = WSF(sim_encoder_dim=hard_hidden_units[-1], com_encoder_dim=hard_hidden_units[-1])
        else:
            self.DFM = sum_fusion(sim_encoder_dim=hard_hidden_units[-1], com_encoder_dim=hard_hidden_units[-1])

    @autocast()
    def forward(self, data):
        sparse_embedding = self.embedding_layer(data)
        sparse_embedding = torch.stack(sparse_embedding, dim=1).squeeze(-2)  # batch,sparse_num(m),emb_dim

        # DNN
        emb_flatten = sparse_embedding.flatten(start_dim=1)

        dnn_input = emb_flatten

        dnn_logit = self.dnn(dnn_input)
        # 一阶交叉
        lr_logit = self.lr(data)

        easy_out, hard_out = self.GM(dnn_input)
        # easy_out = torch.stack(easy_out, dim=1).squeeze(-2)
        # hard_out = torch.stack(hard_out, dim=1).squeeze(-2)
        sim_encoder_out = self.easy_encoder(easy_out)
        hard_flatten = hard_out.flatten(start_dim=1)
        com_encoder_input = hard_flatten
        com_encoder_out = self.hard_encoder(com_encoder_input)

        y_pred2, y_sim, y_com = self.DFM(sim_encoder_out=sim_encoder_out, com_encoder_out=com_encoder_out)

        # CIN
        cin_logit = self.cin(sparse_embedding)


        # # 输出
        # y_pred1 = torch.sigmoid(lr_logit + cin_logit + dnn_logit)
        # y_true = data['label']
        # # loss = self.loss_fun(y_pred.squeeze(-1), data['label'])
        # y_pred = self.output_activation(y_pred1 + y_pred2)
        # loss1 = self.loss_fun(y_pred.squeeze(-1), y_true)
        # y_easy = self.output_activation(y_sim)
        # y_hard = self.output_activation(y_com)
        # tfloss = self.TFLoss(loss_fn=self.loss_fun,y_easy=y_easy.squeeze(-1), y_hard=y_hard.squeeze(-1),
        #                      y_true=y_true, c=self.c,
        #                      gamma=self.gamma, alpha=self.alpha, reduction='mean')
        # 输出
        y_pred1 = torch.sigmoid(lr_logit + cin_logit + dnn_logit)
        y_true = data['label']

        # y_pred = self.output_activation(y_pred1 + y_pred2)
        y_pred = (y_pred1 + y_pred2).squeeze(-1)
        loss1 = self.loss_fun(y_pred, y_true)
        y_easy = y_sim
        y_hard = y_com
        tfloss = self.TFLoss(loss_fn=self.loss_fun, y_easy=y_easy.squeeze(-1), y_hard=y_hard.squeeze(-1),
                             y_true=y_true, c=self.c,
                             gamma=self.gamma, alpha=self.alpha, reduction='mean')
        loss = loss1 + tfloss
        output_dict = {'pred': y_pred, 'loss': loss, 'y_sim': y_sim, 'y_com': y_com,
                       'loss1': loss1, 'tfloss': tfloss
                       }
        return output_dict


    def TFLoss(self,loss_fn, y_easy, y_hard, y_true, c=0.8, gamma=2, alpha=0.25, reduction='mean'):
        assert type is not None, "Missing type parameter. You can choose between easy or hard."
        # y_pred should be 0~1 value
        # EASY LOSS
        Logloss = loss_fn(y_easy, y_true)
        p_t = y_true * y_easy + (1 - y_true) * (1 - y_easy)
        modulating_factor = (c + p_t) ** gamma
        easy_loss = Logloss * modulating_factor
        # HARD LOSS
        Logloss = loss_fn(y_hard, y_true)
        p_t = y_true * y_hard + (1 - y_true) * (1 - y_hard)
        modulating_factor = ((2 - c) - p_t) ** gamma
        hard_loss = Logloss * modulating_factor

        if reduction == 'mean':
            easy_loss = easy_loss.mean()
            hard_loss = hard_loss.mean()
        elif reduction == 'sum':
            easy_loss = easy_loss.sum()
            hard_loss = hard_loss.sum()

        return alpha * easy_loss + (1 - alpha) * hard_loss

# 训练模型，验证模型
def train_model(model, train_loader, optimizer, device, scheduler, metric_list=['roc_auc_score', 'log_loss']):
    model.train()
    pred_list = []
    label_list = []
    pbar = tqdm(train_loader)
    scaler = GradScaler()
    for data in pbar:

        for key in data.keys():
            data[key] = data[key].to(device)
        with autocast():
            output = model(data)
            loss = output['loss']
        pred = output['pred']

        # loss.backward()
        # optimizer.step()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        model.zero_grad()

        pred_list.extend(pred.squeeze(-1).cpu().detach().numpy())
        label_list.extend(data['label'].squeeze(-1).cpu().detach().numpy())
        pbar.set_description("Loss {}".format(loss))

    res_dict = dict()
    res_dict['loss'] = float(output['loss'])
    res_dict['loss1'] = float(output['loss1'])
    res_dict['tfloss'] = float(output['tfloss'])
    for metric in metric_list:
        if metric == 'log_loss':
            res_dict[metric] = log_loss(label_list, pred_list, eps=1e-7)
        else:
            res_dict[metric] = eval(metric)(label_list, pred_list)
    return res_dict


def valid_model(model, valid_loader, device, metric_list=['roc_auc_score', 'log_loss']):
    model.eval()
    pred_list = []
    label_list = []

    for data in (valid_loader):

        for key in data.keys():
            data[key] = data[key].to(device)

        output = model(data)
        pred = output['pred']

        pred_list.extend(pred.squeeze(-1).cpu().detach().numpy())
        label_list.extend(data['label'].squeeze(-1).cpu().detach().numpy())

    res_dict = dict()
    res_dict['loss'] = float(output['loss'])
    res_dict['loss1'] = float(output['loss1'])
    res_dict['tfloss'] = float(output['tfloss'])
    for metric in metric_list:
        if metric == 'log_loss':
            res_dict[metric] = log_loss(label_list, pred_list, eps=1e-7)
        else:
            res_dict[metric] = eval(metric)(label_list, pred_list)

    return res_dict


def test_model(model, test_loader, device, metric_list=['roc_auc_score', 'log_loss']):
    model.eval()
    pred_list = []
    label_list = []

    for data in (test_loader):

        for key in data.keys():
            data[key] = data[key].to(device)

        output = model(data)
        pred = output['pred']

        pred_list.extend(pred.squeeze().cpu().detach().numpy())
        label_list.extend(data['label'].squeeze().cpu().detach().numpy())

    res_dict = dict()
    res_dict['loss'] = float(output['loss'])
    res_dict['loss1'] = float(output['loss1'])
    res_dict['tfloss'] = float(output['tfloss'])
    for metric in metric_list:
        if metric == 'log_loss':
            res_dict[metric] = log_loss(label_list, pred_list, eps=1e-7)
        else:
            res_dict[metric] = eval(metric)(label_list, pred_list)

    return res_dict


# dataloader
train_loader = D.DataLoader(train_dataset, batch_size=config['batch'], shuffle=True, num_workers=config['num_workers'],
                            pin_memory=True)
valid_loader = D.DataLoader(valid_dataset, batch_size=config['batch'], shuffle=False, num_workers=config['num_workers'],
                            pin_memory=True)
test_loader = D.DataLoader(test_dataset, batch_size=config['batch'], shuffle=False, num_workers=config['num_workers'],
                           pin_memory=True)

train_loader = list(train_loader)
valid_loader = list(valid_loader)
test_loader = list(test_loader)


model = xDeepFM(enc_dict=enc_dict,DFM='WSF', embedding_dim=config['embedding_dim'], batch_norm=True, net_dropout=0.4)
device = set_device(config['device'])
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['epoch'], eta_min=0, last_epoch=-1, verbose=False)


model = model.to(device)

performance_metrics = {
    'epoch': [],
    'train_log_loss': [],
    'valid_log_loss': [],
    'test_log_loss': [],

    'train_loss':[],
    'train_loss1':[],
    'train_tfloss':[],

    'valid_loss':[],
    'valid_loss1':[],
    'valid_tfloss':[],

    'test_loss':[],
    'test_loss1':[],
    'test_tfloss':[],

    'train_roc_auc_score':[],
    'valid_roc_auc_score':[],
    'test_roc_auc_score':[]

}
# 模型训练流程
for i in range(1,config['epoch'] + 1):
    # 模型训练
    train_metirc = train_model(model=model, train_loader=train_loader, optimizer=optimizer, device=device,scheduler=scheduler)
    # 模型验证
    valid_metric = valid_model(model, valid_loader, device)
    test_metric = test_model(model, test_loader, device)

    with open('./results/output.txt', 'a') as f:
        # 创建一个新的文件对象，它会把内容写入到我们打开的文件中
        orig_stdout = sys.stdout  # 保存原始的 sys.stdout
        sys.stdout = f  # 重定向 sys.stdout 到文件
        print("epoch:", i, end='  ')
        print("Train Metric:", train_metirc, end='  ')
        print("Valid Metric:", valid_metric,end=' ')

        # 测试模型
        # if i % 5 == 0:
        print('Test Metric:', test_metric)
        sys.stdout = orig_stdout
    print("epoch:", i, end='  ')
    print("Train Metric:", train_metirc, end='  ')
    print("Valid Metric:", valid_metric)

    # 测试模型
    # if i % 5 == 0:
    print('Test Metric:', test_metric)
    # 记录性能指标
    performance_metrics['epoch'].append(i)

    performance_metrics['train_log_loss'].append(train_metirc['log_loss'])
    performance_metrics['train_roc_auc_score'].append(train_metirc['roc_auc_score'])
    performance_metrics['train_loss'].append(train_metirc['loss'])
    performance_metrics['train_loss1'].append(train_metirc['loss1'])
    performance_metrics['train_tfloss'].append(train_metirc['tfloss'])

    performance_metrics['valid_log_loss'].append(valid_metric['log_loss'])
    performance_metrics['valid_roc_auc_score'].append(valid_metric['roc_auc_score'])
    performance_metrics['valid_loss'].append(valid_metric['loss'])
    performance_metrics['valid_loss1'].append(valid_metric['loss1'])
    performance_metrics['valid_tfloss'].append(valid_metric['tfloss'])

    performance_metrics['test_log_loss'].append(test_metric['log_loss'])
    performance_metrics['test_roc_auc_score'].append(test_metric['roc_auc_score'])
    performance_metrics['test_loss'].append(test_metric['loss'])
    performance_metrics['test_loss1'].append(test_metric['loss1'])
    performance_metrics['test_tfloss'].append(test_metric['tfloss'])
    if test_metric['roc_auc_score'] >= 0.8150:
        torch.save(model, './results/' + file_name[:-3] + '_' + str(i) +'_' + str(test_metric['roc_auc_score']) + '.pt')

performance_df = pd.DataFrame.from_dict(performance_metrics)
performance_df.to_csv('./results/'+file_name+'training_performance.csv', index=False)

print("--------模型总的参数量---------")
print(sum(p.numel() for p in model.parameters()))  # 打印模型参数量

print("--------模型训练的参数量---------")
print(sum(p.numel() for p in model.parameters() if p.requires_grad))  # 打印模型参数量