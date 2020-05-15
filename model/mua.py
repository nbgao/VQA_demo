# --------------------------------------------------------
# OpenVQA
# Written by Pengbing Gao https://github.com/nbgao
# --------------------------------------------------------

# from openvqa.ops.fc import FC, MLP
# from openvqa.ops.layer_norm import LayerNorm

import torch.nn as nn
import torch.nn.functional as F
import torch
import math


# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, Cfgs):
        super(MHAtt, self).__init__()
        self.Cfgs = Cfgs

        self.linear_v = nn.Linear(Cfgs.HIDDEN_SIZE, Cfgs.HIDDEN_SIZE)
        self.linear_k = nn.Linear(Cfgs.HIDDEN_SIZE, Cfgs.HIDDEN_SIZE)
        self.linear_q = nn.Linear(Cfgs.HIDDEN_SIZE, Cfgs.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(Cfgs.HIDDEN_SIZE, Cfgs.HIDDEN_SIZE)

        self.dropout = nn.Dropout(Cfgs.DROPOUT_R)

        self.gated_dot_product = GatedDotProduct(Cfgs, dropout_ratio=0)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.Cfgs.MULTI_HEAD,
            int(self.Cfgs.HIDDEN_SIZE / self.Cfgs.MULTI_HEAD)
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.Cfgs.MULTI_HEAD,
            int(self.Cfgs.HIDDEN_SIZE / self.Cfgs.MULTI_HEAD)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.Cfgs.MULTI_HEAD,
            int(self.Cfgs.HIDDEN_SIZE / self.Cfgs.MULTI_HEAD)
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.Cfgs.HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        gate = self.gated_dot_product(key, query)
        key = gate[:, :, :, 0:1] * key
        query = gate[:, :, :, 1:2] * query

        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


# ---------------------------
# ---- Gated Dot-product ----
# ---------------------------

class GatedDotProduct(nn.Module):
    def __init__(self, Cfgs, dropout_ratio=0):
        super(GatedDotProduct, self).__init__()
        self.Cfgs = Cfgs
        self.dropout_ratio = dropout_ratio
        d_base = int(Cfgs.HIDDEN_SIZE / Cfgs.MULTI_HEAD)
        
        self.linearX = nn.Linear(d_base, d_base)
        self.linearY = nn.Linear(d_base, d_base)
        self.linear = nn.Linear(d_base, 2)

        if self.dropout_ratio > 0:
            self.dropout = nn.Dropout(self.dropout_ratio)

    def forward(self, key, query):
        key = self.linearX(key)
        query = self.linearY(query)
        gate = key * query
        
        if self.dropout_ratio > 0:
            gate = self.dropout(gate)
        
        gate = self.linear(gate)
        gate = torch.sigmoid(gate)
        
        return gate


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, Cfgs):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=Cfgs.HIDDEN_SIZE,
            mid_size=Cfgs.FF_SIZE,
            out_size=Cfgs.HIDDEN_SIZE,
            dropout_r=Cfgs.DROPOUT_R,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ---------------------------------
# ---- Unified Attention Block ----
# ---------------------------------

class UA_Block(nn.Module):
    def __init__(self, Cfgs):
        super(UA_Block, self).__init__()

        self.mhatt = MHAtt(Cfgs)
        self.ffn = FFN(Cfgs)

        self.dropout1 = nn.Dropout(Cfgs.DROPOUT_R)
        self.norm1 = LayerNorm(Cfgs.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(Cfgs.DROPOUT_R)
        self.norm2 = LayerNorm(Cfgs.HIDDEN_SIZE)

    def forward(self, x, x_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x


# ------------------------
# ---- Unified Layers ----
# ------------------------

class UnifiedLayers(nn.Module):
    def __init__(self, Cfgs):
        super(UnifiedLayers, self).__init__()
        self.ua_block_list = nn.ModuleList([UA_Block(Cfgs) for _ in range(Cfgs.LAYER)])

    def forward(self, x, mask):
        for ua_block in self.ua_block_list:
            x = ua_block(x, mask)
        return x

# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------

# class MCA_ED(nn.Module):
#     def __init__(self, Cfgs):
#         super(MCA_ED, self).__init__()

#         self.enc_list = nn.ModuleList([SA(Cfgs) for _ in range(Cfgs.LAYER)])
#         self.dec_list = nn.ModuleList([SGA(Cfgs) for _ in range(Cfgs.LAYER)])

#     def forward(self, y, x, y_mask, x_mask):
#         # Get encoder last hidden vector
#         for enc in self.enc_list:
#             y = enc(y, y_mask)

#         # Input encoder last hidden vector
#         # And obtain decoder last hidden vectors
#         for dec in self.dec_list:
#             x = dec(x, y, x_mask, y_mask)

#         return y, x


class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x

class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))

class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2