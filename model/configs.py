import torch.nn as nn

class Cfgs(object):
    # def __init__(self):
    #     super(Cfgs, self).__init__()
        LAYER = 10
        HIDDEN_SIZE = 768
        BBOXFEAT_EMB_SIZE = 2048
        USE_GLOVE = True
        WORD_EMBED_SIZE = 300
        FF_SIZE = 3072
        MULTI_HEAD = 12
        DROPOUT_R = 0.1
        FLAT_MLP_SIZE = 512
        FLAT_GLIMPSES = 1
        FLAT_OUT_SIZE = 1024
        USE_AUX_FEAT = False
        USE_BBOX_FEAT = False
        FEAT_SIZE = {
            'vqa': {
                'FRCN_FEAT_SIZE': (100, 2048),
                'BBOX_FEAT_SIZE': (100, 5),
            },
            'gqa': {
                'FRCN_FEAT_SIZE': (100, 2048),
                'GRID_FEAT_SIZE': (49, 2048),
                'BBOX_FEAT_SIZE': (100, 5),
            },
            'clevr': {
                'GRID_FEAT_SIZE': (196, 1024),
            },
        }