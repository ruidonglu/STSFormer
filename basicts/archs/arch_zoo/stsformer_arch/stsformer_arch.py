import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .util import DataEmbedding, TEncoderBlock, SemanticEmbedding


class STSFormer(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        self.DEVICE = model_args['DEVICE']
        self.in_channels = model_args['in_channels']
        self.adj_mx = model_args['adj_mx']
        self.dropout = model_args['dropout']
        self.alpha = model_args['alpha']
        self.lap_mx = model_args['lap_mx']
        self.kernel_size = model_args['kernel_size']
        self.t_attn_size = model_args['t_attn_size']
        self.t_num_heads = model_args['t_num_heads']
        self.output_size = model_args['output_size']
        self.feature_dim = model_args['feature_dim']
        self.embed_dim = model_args['embed_dim']
        self.time_of_day_size = model_args["time_of_day_size"]
        self.day_of_week_size = model_args["day_of_week_size"]
        self.if_time_of_day = model_args["if_T_o_D"]
        self.if_day_of_week = model_args["if_D_o_W"]
        self.input_dim = model_args['input_dim']
        self.encoder_num = model_args['encoder_num']
        self.temp_dim_tod = model_args['temp_dim_tod']
        self.temp_dim_dow = model_args['temp_dim_dow']
        self.if_spatial = model_args['if_spatial']
        self.num_nodes = model_args['num_nodes']
        self.node_dim = model_args['node_dim']
        self.series_dim = model_args['series_dim']


        self.sem_emd = SemanticEmbedding(self.input_dim, self.num_nodes, self.node_dim, self.temp_dim_tod, self.temp_dim_dow, self.time_of_day_size, self.day_of_week_size, self.if_time_of_day, self.if_day_of_week, self.if_spatial)
        self.padding = (self.kernel_size - 1)//2
        self.Embedding = DataEmbedding(self.feature_dim, self.series_dim, self.lap_mx, self.num_nodes)
        self.T_conv1 = nn.Conv2d(self.embed_dim, self.embed_dim, (1, self.kernel_size), padding=(0, self.padding))
        self.encoder_blocks = nn.ModuleList()
        for _ in range(self.encoder_num):
            self.encoder_blocks.append(TEncoderBlock(self.embed_dim, self.t_attn_size, self.embed_dim, self.kernel_size, self.embed_dim, self.dropout, self.dropout, self.t_num_heads))
        self.generator = nn.Linear(self.embed_dim, self.output_size)


    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        batch_size, num_of_timesteps, num_of_vertices, num_of_features  = history_data.shape
        x = history_data
        sem_emb = self.sem_emd(x)
        x = x.permute(0,2,3,1)
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        
        # Embedding
        # lap_x = get_Laplacian(self.adj_mx)
        x = self.Embedding(x.permute(0,1,3,2)) # (b,n,t,f_in) 
        x = x.permute(0,2,1,3)
        x = torch.cat([x, sem_emb], dim = 3)

        x = x.permute(0,2,3,1)
        x = x.permute(0,2,1,3)  #(b,n,t,f_in)->(b,f_in,n,t)

        x = self.T_conv1(x)   # (b,f_in,n,t)

        for trans in self.encoder_blocks:
            x = trans(x)   # (B,t,N,f)

        x = x.permute(0,3,2,1)
        output = self.generator(x)

        return output
    