'''
@filename: server.py
@author: nbgao (Gao Pengbing)
@contact: nbgao@126.com
'''
# python server.py --gpu=0 --map_location=cpu
import argparse
import os, sys, io, time, json, re
import numpy as np
import cv2
import flask
import torch
import torch.nn
import torch.nn.functional as F
from PIL import Image
from model.net import Net
from model.configs import Cfgs
# from extract_features import extract_feat
from util.translation import process_translate

import logging
sys.path.append('detectron2')
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results

from utils.utils import mkdir, save_features
from utils.extract_utils import get_image_blob
from models import add_config
from models.bua.layers.nms import nms


# Initialize Flask application and the PyTorch model
app = flask.Flask(__name__)
cfg = None
net_img = None
net_vqa = None
token_to_ix = None
ix_to_ans = None

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


def load_model_vqa(map_location):
    global net_vqa
    global token_to_ix
    global ix_to_ans

    # Base path
    model_path = './model/muan.pkl'
    token_to_ix_path = './data/token_to_ix.json'
    pretrained_emb_path = './data/pretrained_emb.npy'
    ans_dict_path = './data/vqa_answer_dict.json'

    '''Pre-load'''
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


    '''Load the pre-trained model'''
    # Load model ckpt
    time_start = time.time()
    print('\nLoading ckpt from: {}'.format(model_path))
    state_dict = torch.load(model_path, map_location=map_location)['state_dict']
    print('state_dict num:', len(state_dict.keys()))
    print('Finish load state_dict!')

    # Load model
    net_vqa = Net(pretrained_emb, token_size, ans_size)
    net_vqa.load_state_dict(state_dict)
    net_vqa.cuda()
    net_vqa.eval()
    # del state_dict
    # print('net:', net)
    time_end = time.time()
    print('Finish load net model!')
    print('Model load time: {:.3f}s\n'.format(time_end-time_start))


# Load image feature extraction model
def setup(args):
    '''
    Create configs and perform basic setups.
    '''
    cfg = get_cfg()
    add_config(args, cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def load_model_img():
    global net_img
    global cfg

    time_start = time.time()
    print('\nLoad image feature extraction model')
    logging.disable()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config-file',
        default='model/extract-bua-caffe-r101.yaml',
        metavar='FILE',
        help='path to config file',
    )
    parser.add_argument('--mode', default='caffe', type=str, help='bua_caffe, ...')
    parser.add_argument(
        '--resume',
        action='store_true',
        help='whether to attempt to resume from the checkpoint directory',
    )
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args([])
    cfg = setup(args)

    MIN_BOXES = 10
    MAX_BOXES = 100
    CONF_THRESH = 0.2

    net_img = DefaultTrainer.build_model(cfg)
    DetectionCheckpointer(net_img, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    net_img.eval()

    time_end = time.time()
    print('Finish load net model!')
    print('Model load time: {:.3f}s\n'.format(time_end-time_start))


def extract_feat(image_path):
    MIN_BOXES = 10
    MAX_BOXES = 100
    CONF_THRESH = 0.2

    im = cv2.imread(image_path)
    print('image shape:', im.shape)
    dataset_dict = get_image_blob(im)

    with torch.set_grad_enabled(False):
        # boxes, scores, features_pooled = model([dataset_dict])
        if cfg.MODEL.BUA.ATTRIBUTE_ON:
            boxes, scores, features_pooled, attr_scores = net_img([dataset_dict])
        else:
            boxes, scores, features_pooled = net_img([dataset_dict])

    dets = boxes[0].tensor.cpu() / dataset_dict['im_scale']
    scores = scores[0].cpu()
    feats = features_pooled[0].cpu()


    max_conf = torch.zeros((scores.shape[0])).to(scores.device)
    for cls_ind in range(1, scores.shape[1]):
            cls_scores = scores[:, cls_ind]
            keep = nms(dets, cls_scores, 0.3)
            max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep],
                                            cls_scores[keep],
                                            max_conf[keep])
        
    keep_boxes = torch.nonzero(max_conf >= CONF_THRESH).flatten()
    if len(keep_boxes) < MIN_BOXES:
        keep_boxes = torch.argsort(max_conf, descending=True)[:MIN_BOXES]
    elif len(keep_boxes) > MAX_BOXES:
        keep_boxes = torch.argsort(max_conf, descending=True)[:MAX_BOXES]
    image_feat = feats[keep_boxes]
    image_bboxes = dets[keep_boxes]
    image_objects_conf = np.max(scores[keep_boxes].numpy(), axis=1)
    image_objects = np.argmax(scores[keep_boxes].numpy(), axis=1)
    if cfg.MODEL.BUA.ATTRIBUTE_ON:
        attr_scores = attr_scores[0].cpu()
        image_attrs_conf = np.max(attr_scores[keep_boxes].numpy(), axis=1)
        image_attrs = np.argmax(attr_scores[keep_boxes].numpy(), axis=1)
        info = {
        'image_id': image_path.split('.')[0],
        'image_h': np.size(im, 0),
        'image_w': np.size(im, 1),
        'num_boxes': len(keep_boxes),
        'objects_id': image_objects,
        'objects_conf': image_objects_conf,
        'attrs_id': image_attrs,
        'attrs_conf': image_attrs_conf,
        }
    else:
        info = {
        'image_id': image_path.split('.')[0],
        'image_h': np.size(im, 0),
        'image_w': np.size(im, 1),
        'num_boxes': len(keep_boxes),
        'objects_id': image_objects,
        'objects_conf': image_objects_conf
        }

    return image_feat, image_bboxes, im.shape[:2]


@app.route("/predict", methods=['POST'])
def predict():
    print('[APP] Predict')
    # Initialize the data dictionary that will be returned from the view.
    result = {"success": False}

    # Ensure an image was properly uploaded to our endpoint.
    if flask.request.method == 'POST':
        data = flask.request.form
        print('data:', data)
        if data['image_path']:
        # if flask.request.files.get('image_path'):
            # Read the image in PIL format
            # image = flask.request.files['image'].read()
            # image = Image.open(io.BytesIO(image))
            image_path = data['image_path']
            question = data['question']
            language = data['language']
            if language in ['ZH', 'zh']:
                chinese_question = question
                print('chinese_question:', chinese_question)
                question, trans_type = process_translate(chinese_question)
            print('image_path:', image_path)
            print('question:', question)
            print('language:', language)

            # process question
            ques_ix = proc_ques(question, token_to_ix, max_token=14)
            ques_ix = torch.from_numpy(ques_ix)
            print('ques_ix:', ques_ix)

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

            # Model predict
            time_start = time.time()
            pred = net_vqa(imgfeat_batch, bboxfeat_batch, quesix_batch)
            pred_np = pred.cpu().data.numpy()
            pred_argmax = np.argmax(pred_np, axis=1)[0]
            pred_ans = ix_to_ans[str(pred_argmax)]
            print('pred_argmax:', pred_argmax)
            print('pred_ans:', pred_ans)
            if language in ['ZH', 'zh']:
                chinese_ans, trans_type = process_translate(pred_ans)
                print('chinese_ans:', chinese_ans)
            else:
                chinese_ans = None
            time_end = time.time()
            print('Predict time: {:.3f}s'.format(time_end-time_start))

            # Return result
            result['predictions'] = list()
            info = {'pred_ans': pred_ans, 'chinese_ans': chinese_ans, 'language': language}
            result['predictions'] = info
            # Indicate that the request was a success
            result['success'] = True

            torch.cuda.empty_cache()

    print('result:', result)
    # Return the data dictionary as a JSON response
    return flask.jsonify(result)


if __name__ == '__main__':
    print('[SERVER]')
    time_start = time.time()
    parser = argparse.ArgumentParser('VQA_demo server')
    parser.add_argument('--gpu', '-gpu', type=str, default='0', choices=['-1', '0','1','2','3'], help='GPU device id')
    parser.add_argument('--map_location', '-map_location', type=str, default='cuda', choices=['cpu', 'cuda'], help='State dict map location')
    args = parser.parse_args()
    gpu_id = args.gpu
    map_location = args.map_location
    print('gpu_id:', gpu_id)
    print('map_location:', map_location)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

    load_model_img()
    load_model_vqa(map_location)
    time_end = time.time()
    print('Server start time: {:.3f}s'.format(time_end-time_start))

    app.run(host='127.0.0.1', port=5000)
    
