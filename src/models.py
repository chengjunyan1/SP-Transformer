# adapted from https://github.com/yaohungt/Multimodal-Transformer/blob/master/src/models.py 
import torch
from torch import nn
import torch.nn.functional as F

from modules.sp_transformer import SPEncoder


class Superformer(nn.Module):
    def __init__(self, hyp_params):
        """
        Construct a Superformer model.
        """
        super(Superformer, self).__init__()
        self.d_model = hyp_params.d_model
        self.input_dims = dict(t=hyp_params.orig_d_l, 
                               a=hyp_params.orig_d_a, 
                               v=hyp_params.orig_d_v)
        self.num_heads = hyp_params.num_heads
        self.layers = hyp_params.layers
        self.attn_dropout = hyp_params.attn_dropout
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.S, self.r = hyp_params.S, hyp_params.r
        self.shift_mode=hyp_params.shift_mode
        self.use_fast=hyp_params.use_fast
        self.use_dense=hyp_params.use_dense
        combined_dim = 3*self.d_model

        self.spe=SPEncoder(embed_dim=self.d_model,
                                input_dims=self.input_dims,
                                num_heads=self.num_heads,
                                layers=self.layers,
                                attn_dropout=self.attn_dropout,
                                relu_dropout=self.relu_dropout,
                                res_dropout=self.res_dropout,
                                embed_dropout=self.embed_dropout,
                                S=self.S, r=self.r,
                                shift_mode=self.shift_mode,
                                use_fast=self.use_fast,
                                use_dense=self.use_dense,
                                device=hyp_params.device)
       
        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, hyp_params.output_dim)

            
    def forward(self,  t, a, v): #[BS,SL,D]
        h_a, h_t, h_v=self.spe(a,t,v)
        last_hs = torch.cat([h_t[-1], h_a[-1], h_v[-1]], dim=1)
        # last_hs = torch.cat([torch.mean(h_t,0), torch.mean(h_a,0), torch.mean(h_v,0)], dim=1)
        # last_hs = self.spe(a,t,v)[-1]

        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        output = self.out_layer(last_hs_proj)
        return output, last_hs