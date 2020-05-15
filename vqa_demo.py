# python vqa_demo.py --mode=INFERENCE --img_path='data/img/test/img0.jpg' --ques='What is on the table?'
# python vqa_demo.py --mode=INFERENCE --img_path='data/img/test/img1.jpg' --ques='这是什么运动?' --lang=ZH
# python vqa_demo.py --mode=INFERENCE --img_path='data/img/test/img2.jpg' --ques='这是什么地方?' --lang=ZH
# python vqa_demo.py --mode=TEST --dataset=test --idx=0
# python vqa_demo.py
import numpy as np
import matplotlib.pyplot as plt
import math, os, shutil, json, pickle, re, time
import matplotlib.pyplot as plt
import argparse
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
# import en_vectors_web_lg
from util.translation import process_translate
from extract_features import extract_feat
from model.net import Net
from model.configs import Cfgs

'''
def tokenize(self, stat_ques_list, use_glove):
    token_to_ix = {
        'PAD': 0,
        'UNK': 1,
        'CLS': 2,
    }

    spacy_tool = None
    pretrained_emb = []
    if use_glove:
        spacy_tool = en_vectors_web_lg.load()
        pretrained_emb.append(spacy_tool('PAD').vector)
        pretrained_emb.append(spacy_tool('UNK').vector)
        pretrained_emb.append(spacy_tool('CLS').vector)

    for ques in stat_ques_list:
        words = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            ques['question'].lower()
        ).replace('-', ' ').replace('/', ' ').split()

        for word in words:
            if word not in token_to_ix:
                token_to_ix[word] = len(token_to_ix)
                if use_glove:
                    pretrained_emb.append(spacy_tool(word).vector)

    pretrained_emb = np.array(pretrained_emb)

    return token_to_ix, pretrained_emb
'''

def proc_ques(ques, token_to_ix, max_token=14):
    ques_ix = np.zeros(max_token, np.int64)
    words = re.sub(r"[.,'!?\"()*#:;]", '', ques.lower()
            ).replace('-', ' ').replace('/', ' ').split()
    print('words:', words)

    for ix, word in enumerate(words):
        if word in token_to_ix:
            ques_ix[ix] = token_to_ix[word]
        else:
            ques_ix[ix] = token_to_ix['UNK']

        if ix == max_token-1:
            break

    return ques_ix

def proc_img_feat(img_feat, img_feat_pad_size):
    img_feat = img_feat[:min(len(img_feat), img_feat_pad_size)]

    img_feat = np.pad(img_feat, ((0, img_feat_pad_size-img_feat.shape[0]), (0,0)),
                mode='constant', constant_values=0)
    return img_feat

def proc_bbox_feat(bbox, img_shape):
    bbox_feat = np.zeros((len(bbox), 5), dtype=np.float32)

    bbox_feat[:, 0] = bbox[:, 0] / img_shape[1]
    bbox_feat[:, 1] = bbox[:, 1] / img_shape[0]
    bbox_feat[:, 2] = bbox[:, 2] / img_shape[1]
    bbox_feat[:, 3] = bbox[:, 3] / img_shape[0]
    bbox_feat[:, 4] = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1]) / (img_shape[0] * img_shape[1])

    return bbox_feat

def copy_image(image_path, feat_path):
    image_target_path = './data/img/' + image_path.split('/')[-1] 
    feat_target_path = './data/feat/' + feat_path.split('/')[-1]
    shutil.copyfile(image_path, image_target_path)
    shutil.copyfile(feat_path, feat_target_path)

def show_image(image_path):
    image = plt.imread(image_path)
    # print(image.shape)
    plt.figure()
    plt.imshow(image)
    # plt.show()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VQA demo')
    parser.add_argument('--mode', '-mode', type=str, default='TEST', choices=['TEST', 'INFERENCE'], help='Running mode')
    parser.add_argument('--dataset', '-dataset', type=str, default='train', choices=['train', 'val', 'test'], help='Dataset type')
    parser.add_argument('--idx', '-idx', type=int, default=0, help='Input image index')
    parser.add_argument('--gpu', '-gpu', type=str, default='0', choices=['0','1','2','3'], help='GPU device id')
    parser.add_argument('--map_location', '-map_location', type=str, default='cuda', choices=['cpu', 'cuda'], help='State dict map location')
    parser.add_argument('--img_path', '-img_path', type=str, default='data/test/img1.jpg', help='image path')
    parser.add_argument('--ques', '-ques', type=str, default='What is the object?', help='input question')
    parser.add_argument('--lang', '-lang', type=str, default='EN', choices=['EN', 'en', 'ZH', 'zh'], help='Question & answer language')
    
    args = parser.parse_args()
    mode = args.mode
    dataset = args.dataset
    idx = args.idx
    gpu_id = args.gpu
    map_location = args.map_location
    language = args.lang
    print('mode:', mode)
    print('dataset:{} idx: {}'.format(dataset, idx))
    print('gpu_id:', gpu_id)
    print('map_location:', map_location)
    print('language:', language)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

    # Base path
    model_path = './model/muan.pkl'
    token_to_ix_path = './data/token_to_ix.json'
    pretrained_emb_path = './data/pretrained_emb.npy'
    ans_dict_path = './data/vqa_answer_dict.json'

    print('\nmodel_path:', model_path)
    print('token_to_ix_path:', token_to_ix_path)
    print('pretrained_emb_path:', pretrained_emb_path)
    print('ans_dict_path:', ans_dict_path)
    print()

    if mode in ['TEST']:
        img_id_list = {
            'train': [570409, 274514, 233146, 143425, 274496, 191651],
            'val': [563194, 432146, 432146, 38886, 563127],
            'test': [353639, 459316, 567500, 527658, 527658, 265455]
        }
        img_id = img_id_list[dataset][idx]
        dataset_dict = {'train': 'train2014', 'val': 'val2014', 'test': 'test2015'}
        dataset_name = dataset_dict[dataset]
        image_path = './data/img/COCO_{}_{:012d}.jpg'.format(dataset_name, img_id)
        frcn_feat_path = './data/feat/COCO_{}_{:012d}.jpg.npz'.format(dataset_name, img_id)
        # frcn_feat_path = '/home/features/vqa/feats/{}/COCO_{}_{:012d}.jpg.npz'.format(dataset_name, dataset_name, img_id)
        # copy_image(image_path, frcn_feat_path)
        # show_image(image_path)
        chinese_ques_list = {
            'train': [
                "行李是什么颜色的?",
                "女人穿的是什么?",
                "这是什么房间?",
                "那个标志是什么意思?",
                "男人的夹克是什么颜色的?",
                "他们玩的运动是什么?",
            ],
            'val': [
                "什么衣服是这个男人正穿着的?",
                "在这张照片里有多少人戴着紫色的帽子?",
                "这个人穿着红衣服吗?",
                "有多少滑雪板运动员?",
                "这种交通工具在空中旅行吗?",
            ],
            'test': [
                "这个人穿的是什么衣服?",
                "有多少人在这张照片中?",
                "桌子上有报纸吗?",
                "人们在哪里?",
                "什么物体是红色的?",
                "墙上是什么?",
            ]
        }
        ques_list = {
            'train':[
                    "What color is the luggage?", 
                    "What is the woman wearing?", 
                    "What room is this?", 
                    "What does that sign mean?", 
                    "What color is the man's jacket?",
                    "Which sport are they playing?", 
            ],
            'val': ["What is the man wearing?",
                    "How many people in this picture are wearing not purple hats?",
                    "Is the man dressed in red?",
                    "How many snowboarders?",
                    "Does this type of transportation travel in the air?",
            ],
            'test': ["What is the person wearing?",
                    "How many people are in the photo?",
                    "Are there newspapers on the table?",
                    "Where are these people?",
                    "What object is red?",
                    "What is on the wall?",
            ]
        }
        if language in ['EN', 'en']:
            question = ques_list[dataset][idx]
        elif language in ['ZH', 'zh']:
            chinese_question = chinese_ques_list[dataset][idx]
            question, trans_type = process_translate(chinese_question)

    elif mode in ['INFERENCE']:
        image_path = args.img_path
        if language in ['EN', 'en']:
            question = args.ques
        elif language in ['ZH', 'zh']:
            chinese_question = args.ques
            question, trans_type = process_translate(chinese_question)

    # Pre-load
    # Load token_to_ix
    token_to_ix = json.load(open(token_to_ix_path, 'r'))
    token_size = len(token_to_ix)
    print(' ========== Question token vocab size:', token_size)
    # print('token_to_ix:', token_to_ix)

    # Load pretrained_emb
    pretrained_emb = np.load(pretrained_emb_path)
    print('pretrained_emb shape:', pretrained_emb.shape)

    # Answers statistic
    ans_to_ix, ix_to_ans = json.load(open(ans_dict_path, 'r'))
    ans_size = len(ans_to_ix)
    print(' ========== Answer token vocab size (occur more than {} times):'.format(8), ans_size)
    # print('ix_to_ans:\n', ix_to_ans)

    # process question
    if language in ['ZH', 'zh']:
        print('chinese question:', chinese_question)
    print('question:', question)
    ques_ix = proc_ques(question, token_to_ix, max_token=14)
    ques_ix = torch.from_numpy(ques_ix)
    print('ques_ix:', ques_ix)

    # frcn_feat.npz:  ['x', 'image_w', 'bbox', 'num_bbox', 'image_h']
    # x: (2048, 54) bbox: (54, 4) num_bbox: 54 image_w: 640 image_h: 480  
    # image_npz = np.load(frcn_feat_path)
    # frcn = image_npz['x'].transpose(1, 0)
    # bbox = image_npz['bbox']
    print('image_path:', image_path)
    time_start = time.time()
    frcn, bbox, image_shape = extract_feat(image_path)

    # padding
    image_feat = proc_img_feat(frcn, Cfgs.FEAT_SIZE['vqa']['FRCN_FEAT_SIZE'][0])
    bbox_feat = proc_bbox_feat(bbox, image_shape)
    image_feat = torch.from_numpy(image_feat)
    bbox_feat = torch.from_numpy(bbox_feat)
    print('image_feat:', image_feat.shape)
    print('bbox_feat:', bbox_feat.shape)

    # batch sample
    quesix_batch = ques_ix.unsqueeze(0).cuda()
    imgfeat_batch = image_feat.unsqueeze(0).cuda()
    bboxfeat_batch = bbox_feat.unsqueeze(0).cuda()
    print('\nquesix_batch:', quesix_batch.shape)
    print('imgfeat_batch:', imgfeat_batch.shape)
    print('bboxfeat_batch:', bboxfeat_batch.shape)

    time_end = time.time()
    print('Image feature process time: {:.3f}s'.format(time_end-time_start))

    # Load model ckpt
    print('\nLoading ckpt from: {}'.format(model_path))
    time_start = time.time()
    state_dict = torch.load(model_path, map_location=map_location)['state_dict']
    print('state_dict num:', len(state_dict.keys()))
    print('Finish load state_dict!')

    # Load model
    net = Net(pretrained_emb, token_size, ans_size)
    net.cuda()
    net.eval()
    net.load_state_dict(state_dict)
    # print('net:', net)
    time_end = time.time()
    print('Finish load net model!')
    print('Model load time: {:.3f}s\n'.format(time_end-time_start))

    # Predict
    time_start = time.time()
    pred = net(imgfeat_batch, bboxfeat_batch, quesix_batch)
    pred_np = pred.cpu().data.numpy()
    pred_argmax = np.argmax(pred_np, axis=1)[0]
    pred_ans = ix_to_ans[str(pred_argmax)]
    print('pred_argmax:', pred_argmax)
    print('pred_ans:', pred_ans)
    if language in ['ZH', 'zh']:
        chinese_ans, trans_type = process_translate(pred_ans)
        print('chinese_ans:', chinese_ans)
    time_end = time.time()
    print('Predict time: {:.3f}s'.format(time_end-time_start))
