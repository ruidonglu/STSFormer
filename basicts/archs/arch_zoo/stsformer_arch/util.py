import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def subsequent_mask(size):
    atten_shape = (1,size,size)
    subsequent_mask = np.triu(np.ones(atten_shape),k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0  


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(x.shape)
        return self.pe[:, :x.size(1)].unsqueeze(2).expand_as(x).detach()
    
class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0., max_len = 100, lookup_index=None):
        super(TemporalPositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.lookup_index = lookup_index
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i+1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0).unsqueeze(0)  # (1, 1, T_max, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, N, T, F_out)
        '''
        if self.lookup_index is not None:
            x = x + self.pe[:, :, self.lookup_index, :]  # (batch_size, N, T, F_in) + (1,1,T,d_model)
        else:
            x = x + self.pe[:, :, :x.size(2), :]

        return self.dropout(x.detach())
    

class LaplacianPE(nn.Module):
    def __init__(self, embed_dim, lape_dim):
        super().__init__()
        self.embedding_lap_pos_enc = nn.Linear(lape_dim, embed_dim)

    def forward(self, lap_mx):
        lap_mx = torch.tensor(lap_mx).to(device='cuda:0')  #(N*N)
        lap_mx = lap_mx.float()
        lap_pos_enc = self.embedding_lap_pos_enc(lap_mx).unsqueeze(0).unsqueeze(0) #(1,1,N,F)
        lap_pos_enc = lap_pos_enc.permute(0,2,1,3)
        return lap_pos_enc

class TokenEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim, norm_layer=None):
        super().__init__()
        self.token_embed = nn.Linear(input_dim, embed_dim, bias=True)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.token_embed(x)
        x = self.norm(x)
        return x
    
class SemanticEmbedding(nn.Module):
    def __init__(self, input_dim, num_nodes, node_dim, temp_dim_tid, temp_dim_diw, time_of_day_size, 
                 day_of_week_size, if_time_of_day, if_day_of_week, if_spatial):
        super().__init__()
    # temporal embeddings
        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.time_of_day_size = time_of_day_size
        self.day_of_week_size = day_of_week_size
        self.temp_dim_tid = temp_dim_tid
        self.temp_dim_diw = temp_dim_diw
        self.if_time_of_day = if_time_of_day
        self.if_day_of_week = if_day_of_week
        self.if_spatial = if_spatial

        # spatial embeddings
        if self.if_spatial:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)

        if self.if_time_of_day:
            self.time_of_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_of_day_emb)
        if self.if_day_of_week:
            self.day_of_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_of_week_emb)

    def forward(self, x):
        batch_size, num_of_timesteps, num_of_vertices, num_of_features = x.shape

        if self.if_time_of_day:
            t_o_d_data = x[..., 1]
            time_of_day_emb = []
            for i in range(12):
                time_of_day_emb.append(self.time_of_day_emb[(t_o_d_data[:, i, :] * self.time_of_day_size).type(torch.LongTensor)])
            time_of_day_emb = torch.stack(time_of_day_emb, dim=3)
        else:
            time_of_day_emb = None
        if self.if_day_of_week:
            d_o_w_data = x[..., 2]
            day_of_week_emb = []
            for i in range(12):
                day_of_week_emb.append(self.day_of_week_emb[(d_o_w_data[:, i, :] * self.day_of_week_size).type(torch.LongTensor)])
            day_of_week_emb = torch.stack(day_of_week_emb, dim=3)
        else:
            day_of_week_emb = None
        
        if self.if_spatial:
            node_emb = []
            for i in range(12):
                node_emb.append(self.node_emb.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2))
            node_emb = torch.stack(node_emb, dim=3)
            
        sem_emb = torch.cat([day_of_week_emb, time_of_day_emb, node_emb.permute(0,2,1,3)],dim=2)
        sem_emb = sem_emb.permute(0,3,1,2)
        return sem_emb
        
    
class DataEmbedding(nn.Module):
    def __init__(
        self, feature_dim, embed_dim, lap_mx, lape_dim,
        drop=0.,device=torch.device('cuda:0'),
    ):
        super().__init__()

        self.device = device
        self.lap_mx = lap_mx
        self.embed_dim = embed_dim
        self.feature_dim = feature_dim
        self.value_embedding = TokenEmbedding(feature_dim, embed_dim)

        self.position_encoding = PositionalEncoding(embed_dim)
        self.temporal_position_encoding = TemporalPositionalEncoding(embed_dim)

        self.spatial_embedding = LaplacianPE(embed_dim, lape_dim)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        origin_x = x
        x = self.value_embedding(origin_x[:, :, :, :self.feature_dim])
        x += self.position_encoding(x)
        x += self.temporal_position_encoding(x)
        # x += self.spatial_embedding(self.lap_mx)

        x = self.dropout(x)
        
        return x


class TemporalSelfAttention(nn.Module):
    def __init__(
        self, dim, dim_out, t_attn_size, attn_drop, proj_drop, 
        t_num_heads, qkv_bias=False, device=torch.device('cuda:0')
    ):
        super().__init__()
        # assert dim % t_num_heads == 0
        self.t_num_heads = t_num_heads
        self.head_dim = dim // t_num_heads
        self.scale = self.head_dim ** -0.5
        self.device = device
        self.t_attn_size = t_attn_size
        self.t_num_heads = t_num_heads

        self.t_q_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.t_k_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.t_v_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.t_attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):

        B, T, N, D = x.shape

        t_q = self.t_q_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_k = self.t_k_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_v = self.t_v_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)

        t_q = t_q.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_k = t_k.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_v = t_v.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)

        t_attn = (t_q @ t_k.transpose(-2, -1)) * self.scale

        t_attn = t_attn.softmax(dim=-1)
        t_attn = self.t_attn_drop(t_attn)

        t_x = (t_attn @ t_v).transpose(2, 3).reshape(B, N, T, D).transpose(1, 2)

        x = self.proj(t_x)
        x = self.proj_drop(x)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class TEncoderBlock(nn.Module):
    def __init__(
        self, dim, t_attn_size, embed_dim, kernel_size, output_dim, drop, attn_drop, t_num_heads, mlp_ratio=4., qkv_bias=True, onv_layer=nn.Conv2d,
        drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, device=torch.device('cuda:0'), type_ln="pre",
    ):
        super().__init__()
        self.type_ln = type_ln
        self.padding = (kernel_size - 1)//2
        self.norm1 = norm_layer(dim)
        self.T_conv1 = nn.Conv2d(embed_dim, embed_dim, (1, kernel_size), padding=(0, self.padding))

        self.t_attn = TemporalSelfAttention(
            dim, output_dim, t_attn_size, t_num_heads=t_num_heads, attn_drop=attn_drop, 
            proj_drop=drop,qkv_bias=qkv_bias, device=device, 
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x.permute(0,3,2,1)

        if self.type_ln == 'pre':
            x = x + self.drop_path(self.t_attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        elif self.type_ln == 'post':
            x = self.norm1(x + self.drop_path(self.t_attn(x)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        return x.permute(0,3,2,1)
    