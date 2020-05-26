'''
@filename: client.py
@author: nbgao (Gao Pengbing)
@contact: nbgao@126.com
'''
# python client.py --mode=TEST --dataset=train --idx=2
# python client.py --mode=INFERENCE --img_path='data/img/test/img0.jpg' --ques="What's on the table?"
# python client.py --mode=INFERENCE --img_path='data/img/test/img1.jpg' --ques="这是什么运动?" --lang=ZH
# python client.py --mode=INFERENCE --img_path='data/img/test/img2.jpg' --ques="Where is it?" --lang=EN
import requests
import argparse
# from util.translation import process_translate

# Initialize the PyTorch REST API endpoint URL
REST_API_URL = 'http://127.0.0.1:5000/predict'

def predict_result(image_path, question, language):
    # Initialize image path
    # image = open(image_path, 'rb').read()
    # payload = {'image': image}
    payload = {'image_path': image_path, 'question': question, 'language': language}

    # Submit the request
    # r = requests.post(REST_API_URL, files=payload).json()
    r = requests.post(REST_API_URL, data=payload).json()
    # Return predict result
    # Ensure the request was successful
    if r['success']:
        # Loop over the predictions and display
        result = r['predictions']
        pred_ans = result['pred_ans']
        chinese_ans = result['chinese_ans']
        language = result['language']
        print('pred_ans:', pred_ans)
        if language in ['ZH', 'zh']:
            print('chinese_ans:', chinese_ans)

    else:
        print('Request failed')


if __name__ == '__main__':
    '''
    mode: INFERENCE: 服务器线上推理模式  TEST: 测试模式
    dataset: TEST模式下数据集名称
    idx: 数据集内索引号
    img_path: INFERENCE模式下图片路径
    ques: INFERENCE下问题
    lang: 问题/答案语言选项
    '''
    parser = argparse.ArgumentParser(description='VQA_demo client')
    parser.add_argument('--mode', '-mode', type=str, default='INFERENCE', choices=['TEST', 'test', 'INFERENCE', 'inference'], help='Running mode')
    parser.add_argument('--dataset', '-dataset', type=str, default='train', choices=['train', 'val', 'test'], help='Dataset type')
    parser.add_argument('--idx', '-idx', type=int, default=0, help='Input image index')
    parser.add_argument('--img_path', '-img_path', type=str, default='data/test/img1.jpg', help='image path')
    parser.add_argument('--ques', '-ques', type=str, default='What is the object?', help='input question')
    parser.add_argument('--lang', '-lang', type=str, default='EN', choices=['EN', 'en', 'ZH', 'zh'], help='Question & answer language')
    
    args = parser.parse_args()
    mode = args.mode
    # image_path = args.img_path
    # question = args.ques
    language = args.lang
    print('[CLIENT]')
    print('mode:', mode)
    print('language:', language)

    if mode in ['TEST', 'test']:
        img_id_list = {
            'train': [570409, 274514, 233146, 143425, 274496, 191651],
            'val': [563194, 432146, 432146, 38886, 563127],
            'test': [353639, 459316, 567500, 527658, 527658, 265455]
        }
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

        dataset = args.dataset
        idx = args.idx
        print('dataset:{} idx: {}'.format(dataset, idx))

        img_id = img_id_list[dataset][idx]
        dataset_dict = {'train': 'train2014', 'val': 'val2014', 'test': 'test2015'}
        dataset_name = dataset_dict[dataset]
        image_path = './data/img/COCO_{}_{:012d}.jpg'.format(dataset_name, img_id)
        # frcn_feat_path = './data/feat/COCO_{}_{:012d}.jpg.npz'.format(dataset_name, img_id)
        
        if language in ['EN', 'en']:
            question = ques_list[dataset][idx]
        elif language in ['ZH', 'zh']:
            question = chinese_ques_list[dataset][idx]
            # chinese_question = chinese_ques_list[dataset][idx]
            # question, trans_type = process_translate(chinese_question)

    elif mode in ['INFERENCE', 'inference']:
        image_path = args.img_path
        question = args.ques
        # if language in ['EN', 'en']:
        #     question = args.ques
        # elif language in ['ZH', 'zh']:
        #     chinese_question = args.ques
            # question, trans_type = process_translate(chinese_question)

    # image_path = args.image_path
    print('image_path:', image_path)
    # if language in ['ZH', 'zh']:
    #     print('chinese_quesion:', chinese_question)
    print('question:', question)

    predict_result(image_path, question, language)
