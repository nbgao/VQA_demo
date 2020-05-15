# --------------------------------------------------------
# OpenVQA
# Written by Pengbing Gao https://github.com/nbgao
# --------------------------------------------------------

import torch.nn as nn
import torch
# from model.configs import Cfgs
# from openvqa.core.base_dataset import BaseAdapter
# from openvqa.utils.make_mask import make_mask

# FEAT_SIZE = {
#     'vqa': {
#         'FRCN_FEAT_SIZE': (100, 2048),
#         'BBOX_FEAT_SIZE': (100, 5),
#     },
#     'gqa': {
#         'FRCN_FEAT_SIZE': (100, 2048),
#         'GRID_FEAT_SIZE': (49, 2048),
#         'BBOX_FEAT_SIZE': (100, 5),
#     },
#     'clevr': {
#         'GRID_FEAT_SIZE': (196, 1024),
#     },
# }
# WORD_EMBED_SIZE = 300
# USE_BBOX_FEAT = False
# BBOXFEAT_EMB_SIZE = 2048

class Adapter(nn.Module):
    def __init__(self, Cfgs):
        super(Adapter, self).__init__()
        self.Cfgs = Cfgs
        imgfeat_linear_size = Cfgs.FEAT_SIZE['vqa']['FRCN_FEAT_SIZE'][1]
        if Cfgs.USE_BBOX_FEAT:
            self.bbox_linear = nn.Linear(5, Cfgs.BBOXFEAT_EMB_SIZE)
            imgfeat_linear_size += Cfgs.BBOXFEAT_EMB_SIZE
        self.frcn_linear = nn.Linear(imgfeat_linear_size, Cfgs.HIDDEN_SIZE)

    def forward(self, frcn_feat, bbox_feat):
        img_feat_mask = make_mask(frcn_feat)

        if self.Cfgs.USE_BBOX_FEAT:
            bbox_feat = self.bbox_linear(bbox_feat)
            frcn_feat = torch.cat((frcn_feat, bbox_feat), dim=-1)
        img_feat = self.frcn_linear(frcn_feat)

        return img_feat, img_feat_mask

    # def gqa_init(self, __C):
    #     imgfeat_linear_size = __C.FEAT_SIZE['gqa']['FRCN_FEAT_SIZE'][1]
    #     if __C.USE_BBOX_FEAT:
    #         self.bbox_linear = nn.Linear(5, __C.BBOXFEAT_EMB_SIZE)
    #         imgfeat_linear_size += __C.BBOXFEAT_EMB_SIZE
    #     self.frcn_linear = nn.Linear(imgfeat_linear_size, __C.HIDDEN_SIZE)

    #     if __C.USE_AUX_FEAT:
    #         self.grid_linear = nn.Linear(__C.FEAT_SIZE['gqa']['GRID_FEAT_SIZE'][1], __C.HIDDEN_SIZE)


    # def clevr_init(self, __C):
    #     self.grid_linear = nn.Linear(__C.FEAT_SIZE['clevr']['GRID_FEAT_SIZE'][1], __C.HIDDEN_SIZE)


    # def vqa_forward(self, feat_dict):
    #     frcn_feat = feat_dict['FRCN_FEAT']
    #     bbox_feat = feat_dict['BBOX_FEAT']

    #     img_feat_mask = make_mask(frcn_feat)

    #     if USE_BBOX_FEAT:
    #         bbox_feat = self.bbox_linear(bbox_feat)
    #         frcn_feat = torch.cat((frcn_feat, bbox_feat), dim=-1)
    #     img_feat = self.frcn_linear(frcn_feat)

    #     return img_feat, img_feat_mask


    # def gqa_forward(self, feat_dict):
    #     frcn_feat = feat_dict['FRCN_FEAT']
    #     bbox_feat = feat_dict['BBOX_FEAT']
    #     grid_feat = feat_dict['GRID_FEAT']

    #     img_feat_mask = make_mask(frcn_feat)

    #     if self.__C.USE_BBOX_FEAT:
    #         bbox_feat = self.bbox_linear(bbox_feat)
    #         frcn_feat = torch.cat((frcn_feat, bbox_feat), dim=-1)
    #     img_feat = self.frcn_linear(frcn_feat)

    #     if self.__C.USE_AUX_FEAT:
    #         grid_feat_mask = make_mask(grid_feat)
    #         img_feat_mask = torch.cat((img_feat_mask, grid_feat_mask), dim=-1)
    #         grid_feat = self.grid_linear(grid_feat)
    #         img_feat = torch.cat((img_feat, grid_feat), dim=1)

    #     return img_feat, img_feat_mask


    # def clevr_forward(self, feat_dict):
    #     grid_feat = feat_dict['GRID_FEAT']

    #     img_feat_mask = make_mask(grid_feat)
    #     img_feat = self.grid_linear(grid_feat)

    #     return img_feat, img_feat_mask

# Masking the sequence mask
def make_mask(feature):
    return (torch.sum(
        torch.abs(feature),
        dim=-1
    ) == 0).unsqueeze(1).unsqueeze(2)

