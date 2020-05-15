# --------------------------------------------------------
# OpenVQA
# Written by Pengbing Gao https://github.com/nbgao
# --------------------------------------------------------

# from openvqa.utils.make_mask import make_mask
# from openvqa.ops.fc import FC, MLP
# from openvqa.ops.layer_norm import LayerNorm
from model.mua import UnifiedLayers
from model.adapter import Adapter
from model.configs import Cfgs

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import en_vectors_web_lg

# ------------------------------
# ---- Flatten the sequence ----
# ------------------------------

class TokenFlat(nn.Module):
    def __init__(self, token_pos=0):
        super(TokenFlat, self).__init__()
        self.token_pos = token_pos
    
    def forward(self, x, x_mask):
        return x[:, self.token_pos, :]

'''
class AttFlat(nn.Module):
    def __init__(self, Cfgs):
        super(AttFlat, self).__init__()
        self.Cfgs = Cfgs

        self.mlp = MLP(
            in_size=Cfgs.HIDDEN_SIZE,
            mid_size=Cfgs.FLAT_MLP_SIZE,
            out_size=Cfgs.FLAT_GLIMPSES,
            dropout_r=Cfgs.DROPOUT_R,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            Cfgs.HIDDEN_SIZE * Cfgs.FLAT_GLIMPSES,
            Cfgs.FLAT_OUT_SIZE
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.Cfgs.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted
'''

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

# -------------------------
# ---- Main MUAN Model ----
# -------------------------

class Net(nn.Module):
    def __init__(self, pretrained_emb, token_size, answer_size):
        super(Net, self).__init__()
        self.Cfgs = Cfgs

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=Cfgs.WORD_EMBED_SIZE
        )

        # spacy_tool = en_vectors_web_lg.load()
        # cls_vector = np.expand_dims(spacy_tool('CLS').vector, axis=0)
        # pretrained_emb = np.concatenate((cls_vector, pretrained_emb), axis=0)

        # Loading the GloVe embedding weights
        if Cfgs.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.LSTM(
            input_size=Cfgs.WORD_EMBED_SIZE,
            hidden_size=Cfgs.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.adapter = Adapter(Cfgs)

        self.norm1 = LayerNorm(Cfgs.HIDDEN_SIZE)
        self.norm2 = LayerNorm(Cfgs.HIDDEN_SIZE)

        self.backbone = UnifiedLayers(Cfgs)

        # Flatten to vector
        self.flat = TokenFlat(token_pos=0)

        # Classification layers
        # self.proj_norm = LayerNorm(Cfgs.HIDDEN_SIZE)
        self.proj = nn.Linear(Cfgs.HIDDEN_SIZE, answer_size)


    # def forward(self, frcn_feat, grid_feat, bbox_feat, ques_ix):
    def forward(self, frcn_feat, bbox_feat, ques_ix):
        cls_tensor = torch.full((ques_ix.shape[0], 1), 2, dtype=torch.long).cuda()
        ques_ix = torch.cat((cls_tensor, ques_ix), dim=1)

        # Pre-process Language Feature
        lang_feat_mask = make_mask(ques_ix.unsqueeze(2))
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.lstm(lang_feat)

        # img_feat, img_feat_mask = self.adapter(frcn_feat, grid_feat, bbox_feat)
        img_feat, img_feat_mask = self.adapter(frcn_feat, bbox_feat)

        lang_feat = self.norm1(lang_feat)
        img_feat = self.norm2(img_feat)

        fuse_feat = torch.cat((lang_feat, img_feat), dim=1)
        fuse_feat_mask = torch.cat((lang_feat_mask, img_feat_mask), dim=-1)

        # Backbone Framework
        fuse_feat = self.backbone(fuse_feat, fuse_feat_mask)

        # Flatten to vector
        fuse_flat = self.flat(
            fuse_feat,
            fuse_feat_mask
        )

        # Classification layers
        # proj_feat = lang_feat + img_feat
        # proj_feat = self.proj_norm(proj_feat)
        proj_feat = self.proj(fuse_flat)

        return proj_feat

# Masking the sequence mask
def make_mask(feature):
    return (torch.sum(
        torch.abs(feature),
        dim=-1
    ) == 0).unsqueeze(1).unsqueeze(2)