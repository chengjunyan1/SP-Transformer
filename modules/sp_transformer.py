import torch
from torch import nn
import torch.nn.functional as F
from modules.position_embedding import SinusoidalPositionalEmbedding
from modules.fast_attention import Attention
from modules.multihead_attention import MultiheadAttention
import math


class SPEncoder(nn.Module):

    def __init__(self, input_dims, embed_dim, num_heads, layers, attn_dropout=0.0, relu_dropout=0.0, 
                 res_dropout=0.0, embed_dropout=0.0, S=1, r=[1,1,1], 
                 shift_mode=dict(I=['S','P','R'],X=['S'],S=['S'],C=[1,0.25,0.05]),
                 use_fast=False,use_dense=False,device='cuda'):
        super().__init__()
        self.dropout = embed_dropout # Embedding dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)
        self.init_hiddens=nn.Parameter(torch.Tensor(3, embed_dim))
        nn.init.xavier_uniform_(self.init_hiddens)
        self.shift_mode=shift_mode
        self.use_fast=use_fast
        self.use_dense=use_dense
        self.device=device

        if self.use_fast:
            self.proj_a = nn.Conv1d(input_dims['a'], self.embed_dim, kernel_size=1)
            self.proj_t = nn.Conv1d(input_dims['t'], self.embed_dim, kernel_size=1)
            self.proj_v = nn.Conv1d(input_dims['v'], self.embed_dim, kernel_size=1)
            input_dims=dict(a=self.embed_dim,t=self.embed_dim,v=self.embed_dim)

        self.layers,self.stride=layers,S
        self.hiddenlayer = SPEncoderHiddenLayer(embed_dim,
                                    input_dims,
                                    num_heads=num_heads,
                                    attn_dropout=attn_dropout,
                                    relu_dropout=relu_dropout,
                                    res_dropout=res_dropout,
                                    stride=S,
                                    pf_hidden=r[0]*2+1,
                                    use_fast=self.use_fast,
                                    use_dense=self.use_dense,
                                    device=self.device)
        self.crosslayer = SPEncoderCrossLayer(embed_dim,
                                    input_dims,
                                    num_heads=num_heads,
                                    attn_dropout=attn_dropout,
                                    relu_dropout=relu_dropout,
                                    res_dropout=res_dropout,
                                    pf_cross=r[1]*2+1,
                                    use_fast=self.use_fast,
                                    use_dense=self.use_dense,
                                    device=self.device)
        self.selflayer = SPEncoderSelfLayer(embed_dim,
                                    input_dims,
                                    num_heads=num_heads,
                                    attn_dropout=attn_dropout,
                                    relu_dropout=relu_dropout,
                                    res_dropout=res_dropout,
                                    pf_self=r[2]*2+1,
                                    use_fast=self.use_fast,
                                    use_dense=self.use_dense,
                                    device=self.device)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def init_emb(self,x):
        x = self.embed_scale * x
        if self.embed_positions is not None:
            x += self.embed_positions(x.transpose(0, 1)[:, :, 0],x.shape[-1]).transpose(0, 1)   # Add positional embedding
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, a, t, v): # [BS, SL, D]
        sla,slt,slv,bs=a.shape[1],t.shape[1],v.shape[1],a.shape[0]
        hla,hlt,hlv=math.ceil(sla/self.stride),math.ceil(slt/self.stride),math.ceil(slv/self.stride)
        h_a=self.init_hiddens[0,:].unsqueeze(0).repeat(hla,bs,1)
        h_t=self.init_hiddens[1,:].unsqueeze(0).repeat(hlt,bs,1)
        h_v=self.init_hiddens[2,:].unsqueeze(0).repeat(hlv,bs,1)

        # embed tokens and positions 
        if self.use_fast:
            a=self.proj_a(a.transpose(1, 2)).permute(2, 0, 1)
            t=self.proj_t(t.transpose(1, 2)).permute(2, 0, 1)
            v=self.proj_v(v.transpose(1, 2)).permute(2, 0, 1)
        else: a,t,v=a.permute(1,0,2),t.permute(1,0,2),v.permute(1,0,2)
        a,t,v=self.init_emb(a),self.init_emb(t),self.init_emb(v)
        h_a,h_t,h_v=self.init_emb(h_a),self.init_emb(h_t),self.init_emb(h_v)

        # encoder layers
        shift_mode=self.shift_mode
        for i in range(self.layers):
            shift_a,shift_t,shift_v=0,0,0
            if 'S' in shift_mode['I']: 
                A=shift_mode['C'][0]
                shift_a+=i*A; shift_t+=i*A; shift_v+=i*A
            if 'P' in shift_mode['I']:
                B=shift_mode['C'][1]
                shift_a+=(sla*torch.sin(torch.arange(hla)*B)).long().to(self.device)
                shift_t+=(slt*torch.sin(torch.arange(hlt)*B)).long().to(self.device)
                shift_v+=(slv*torch.sin(torch.arange(hlv)*B)).long().to(self.device)
            if 'R' in shift_mode['I']:
                G=shift_mode['C'][2]
                shift_a+=torch.randint(0,math.ceil(G*sla/hla),[hla],device=self.device)
                shift_t+=torch.randint(0,math.ceil(G*slt/hlt),[hlt],device=self.device)
                shift_v+=torch.randint(0,math.ceil(G*slv/hlt),[hlv],device=self.device)
            h_a, h_t, h_v=self.hiddenlayer(a,t,v,h_a,h_t,h_v,shift_a,shift_t,shift_v)

            shift_a,shift_t,shift_v=0,0,0
            if 'S' in shift_mode['X']: 
                A=shift_mode['C'][0]
                shift_a+=i*A; shift_t+=i*A; shift_v+=i*A
            if 'P' in shift_mode['X']:
                B=shift_mode['C'][1]
                shift_a+=(sla*torch.sin(torch.arange(hla)*B)).long().to(self.device)
                shift_t+=(slt*torch.sin(torch.arange(hlt)*B)).long().to(self.device)
                shift_v+=(slv*torch.sin(torch.arange(hlv)*B)).long().to(self.device)
            if 'R' in shift_mode['X']:
                G=shift_mode['C'][2]
                shift_a+=torch.randint(0,math.ceil(G*sla/hla),[hla],device=self.device)
                shift_t+=torch.randint(0,math.ceil(G*slt/hlt),[hlt],device=self.device)
                shift_v+=torch.randint(0,math.ceil(G*slv/hlt),[hlv],device=self.device)
            h_a, h_t, h_v=self.crosslayer(h_a,h_t,h_v,shift_a,shift_t,shift_v)

            shift_a,shift_t,shift_v=0,0,0
            if 'S' in shift_mode['S']: 
                A=shift_mode['C'][0]
                shift_a+=i*A; shift_t+=i*A; shift_v+=i*A
            if 'P' in shift_mode['S']:
                B=shift_mode['C'][1]
                shift_a+=(sla*torch.sin(torch.arange(hla)*B)).long().to(self.device)
                shift_t+=(slt*torch.sin(torch.arange(hlt)*B)).long().to(self.device)
                shift_v+=(slv*torch.sin(torch.arange(hlv)*B)).long().to(self.device)
            if 'R' in shift_mode['S']:
                G=shift_mode['C'][2]
                shift_a+=torch.randint(0,math.ceil(G*sla/hla),[hla],device=self.device)
                shift_t+=torch.randint(0,math.ceil(G*slt/hlt),[hlt],device=self.device)
                shift_v+=torch.randint(0,math.ceil(G*slv/hlt),[hlv],device=self.device)
            h_a, h_t, h_v=self.selflayer(h_a,h_t,h_v,shift_a,shift_t,shift_v)

        h_a = self.layer_norm(h_a)
        h_t = self.layer_norm(h_t)
        h_v = self.layer_norm(h_v)
        return h_a,h_t,h_v

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())


class FFN(nn.Module):
    def __init__(self, embed_dim,relu_dropout,res_dropout):
        super().__init__()
        self.embed_dim=embed_dim
        self.relu_dropout=relu_dropout
        self.res_dropout=res_dropout
        self.fc1 = Linear(self.embed_dim, 4*self.embed_dim)   # The "Add & Norm" part in the paper
        self.fc2 = Linear(4*self.embed_dim, self.embed_dim)
        self.layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self,x):
        residual = x
        x = self.layer_norm(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        return x


class SPF(nn.Module): # Sparse Phased Fast attention
    def __init__(self,embed_dim,num_heads,attn_dropout,res_dropout,input_dim=None,stride=1,pf=None,
                 generalized_attention=False,dim_head_down=1,use_fast=True,use_dense=False,device='cuda',):
        super().__init__()
        self.embed_dim=embed_dim
        self.num_heads=num_heads
        self.res_dropout=res_dropout
        self.stride, self.pf = stride, pf
        self.generalized_attention=generalized_attention
        self.dim_head=int(embed_dim/num_heads/dim_head_down)
        self.use_fast=use_fast
        self.use_dense=use_dense

        if not use_dense and use_fast:
            self.attn = Attention(
                self.embed_dim,
                heads = self.num_heads,
                dim_head = self.dim_head,
                generalized_attention = self.generalized_attention
            )
        else:
            self.attn = MultiheadAttention(self.embed_dim, self.num_heads, input_dim=input_dim)
        self.layer_norm = nn.LayerNorm(self.embed_dim)
        input_dim=self.embed_dim if input_dim is None else input_dim
        self.layer_norm_kv = nn.LayerNorm(input_dim)
        self.device=device

    def forward(self,x,x_k=None,x_v=None,shift=0,reverse=False):
        sl,bs,_=x.shape
        residual = x
        x = self.layer_norm(x)
        context=x if x_k is None else x_k
        if self.use_dense:
            mask=sparse_mask(x, context, self.stride, self.pf); c=context
        else:
            fetch=sparsify(x, context, self.stride, self.pf, shift, self.device)
            x=x.unsqueeze(2).reshape(-1,1,self.embed_dim)
            c=fetch.permute(1,2,0,3).reshape(-1,self.pf,fetch.shape[-1])
            if not self.use_fast: x=x.permute(1,0,2); c=c.permute(1,0,2)
        if x_k is not None: c = self.layer_norm_kv(c)
        if self.use_dense:
            x,_ = self.attn(x, c, c, reverse=reverse, attn_mask=mask)
        else:
            if self.use_fast: x = self.attn(x, context=c, reverse=reverse)
            else: x,_ = self.attn(x, c, c, reverse=reverse)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = x.squeeze(1).reshape(sl,bs,-1)
        x = residual + x
        return x


class SPEncoderHiddenLayer(nn.Module):
    def __init__(self, embed_dim, input_dims, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, 
                    res_dropout=0.1, stride=None, pf_hidden=None, use_fast=False, use_dense=False, device='cuda'):
        super().__init__()
        self.mha_a=SPF(embed_dim,num_heads,attn_dropout,res_dropout,input_dims['a'],stride,pf_hidden,use_fast=use_fast,use_dense=use_dense,device=device)
        self.ffn_a=FFN(embed_dim,relu_dropout,res_dropout)
        self.mha_t=SPF(embed_dim,num_heads,attn_dropout,res_dropout,input_dims['t'],stride,pf_hidden,use_fast=use_fast,use_dense=use_dense,device=device)
        self.ffn_t=FFN(embed_dim,relu_dropout,res_dropout)
        self.mha_v=SPF(embed_dim,num_heads,attn_dropout,res_dropout,input_dims['v'],stride,pf_hidden,use_fast=use_fast,use_dense=use_dense,device=device)
        self.ffn_v=FFN(embed_dim,relu_dropout,res_dropout)

    def forward(self, a, t, v, h_a, h_t, h_v,shift_a=0,shift_t=0,shift_v=0):
        h_a=self.ffn_a(self.mha_a(h_a,a,a,shift_a))
        h_t=self.ffn_t(self.mha_t(h_t,t,t,shift_t))
        h_v=self.ffn_v(self.mha_v(h_v,v,v,shift_v))
        return h_a, h_t, h_v

def sum_fuse(x,y): return x+y

class SPEncoderCrossLayer(nn.Module):
    def __init__(self, embed_dim, input_dims, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, 
                    res_dropout=0.1, pf_cross=None, use_fast=False, use_dense=False, device='cuda'):
        super().__init__()
        self.mha_at=SPF(embed_dim,num_heads,attn_dropout,res_dropout,pf=pf_cross,use_fast=use_fast,use_dense=use_dense,device=device)
        self.mha_tv=SPF(embed_dim,num_heads,attn_dropout,res_dropout,pf=pf_cross,use_fast=use_fast,use_dense=use_dense,device=device)
        self.mha_va=SPF(embed_dim,num_heads,attn_dropout,res_dropout,pf=pf_cross,use_fast=use_fast,use_dense=use_dense,device=device)
        self.ffn_at=FFN(embed_dim,relu_dropout,res_dropout)
        self.ffn_tv=FFN(embed_dim,relu_dropout,res_dropout)
        self.ffn_va=FFN(embed_dim,relu_dropout,res_dropout)
        self.fuse_a=self.fuse_t=self.fuse_v=sum_fuse

    def forward(self, h_a, h_t, h_v,shift_a=0,shift_t=0,shift_v=0):
        h_at=self.ffn_at(self.mha_at(h_a,h_t,h_t,shift_a))
        h_tv=self.ffn_tv(self.mha_tv(h_t,h_v,h_v,shift_t))
        h_va=self.ffn_va(self.mha_va(h_v,h_a,h_a,shift_v))
        h_ta=self.ffn_at(self.mha_at(h_t,h_a,h_a,shift_t,True))
        h_vt=self.ffn_tv(self.mha_tv(h_v,h_t,h_t,shift_v,True))
        h_av=self.ffn_va(self.mha_va(h_a,h_v,h_v,shift_a,True))
        return self.fuse_a(h_at,h_av), self.fuse_t(h_ta,h_tv), self.fuse_v(h_va,h_vt)


class SPEncoderSelfLayer(nn.Module):
    def __init__(self, embed_dim, input_dims, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, 
                 res_dropout=0.1, pf_self=None, use_fast=False, use_dense=False, device='cuda'):
        super().__init__()
        self.embed_dim=embed_dim
        self.mha_a=SPF(embed_dim,num_heads,attn_dropout,res_dropout,pf=pf_self,use_fast=use_fast,use_dense=use_dense,device=device)
        self.ffn_a=FFN(embed_dim,relu_dropout,res_dropout)
        self.mha_t=SPF(embed_dim,num_heads,attn_dropout,res_dropout,pf=pf_self,use_fast=use_fast,use_dense=use_dense,device=device)
        self.ffn_t=FFN(embed_dim,relu_dropout,res_dropout)
        self.mha_v=SPF(embed_dim,num_heads,attn_dropout,res_dropout,pf=pf_self,use_fast=use_fast,use_dense=use_dense,device=device)
        self.ffn_v=FFN(embed_dim,relu_dropout,res_dropout)

    def forward(self, h_a, h_t, h_v,shift_a=0,shift_t=0,shift_v=0):
        h_a=self.ffn_a(self.mha_a(h_a,shift=shift_a))
        h_t=self.ffn_t(self.mha_t(h_t,shift=shift_t))
        h_v=self.ffn_v(self.mha_v(h_v,shift=shift_v))
        return h_a, h_t, h_v


def sparsify(hidden,context,stride,pf,shift=0,device='cuda'):
    h,bs,_ = hidden.shape
    c,_,dc = context.shape
    r=(torch.arange(h).to(device)+1)*stride-pf+shift
    r=r.unsqueeze(0).repeat(pf,1)+torch.arange(pf).unsqueeze(1).to(device)
    r=r.reshape([pf*h])
    return context[r%c].reshape(pf,h,bs,dc)
    
def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias: nn.init.constant_(m.bias, 0.)
    return m

def sparse_mask(hidden=None,context=None,stride=None,
    pf=None,h=None,c=None,cuda=None,shift=0): #generate 
    h=hidden.size(0) if h is None else h
    c=context.size(0) if c is None else c
    mask=torch.ones(h,c)*torch.tensor(float('-inf'))
    for i in range(pf): 
        k=(torch.arange(h)+1)*stride-pf+i+shift
        mask[torch.arange(h),k%c]=0
    if cuda or (context is not None and context.is_cuda): mask = mask.cuda()
    return mask