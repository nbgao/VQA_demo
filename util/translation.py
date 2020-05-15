import json
import requests

def translate(word):
    url = 'http://fanyi.youdao.com/translate?smartresult=dict&smartresult=rule&smartresult=ugc&sessionFrom=null'
    # url = ' http://fanyi.youdao.com/'
    key = {
        'type': 'AUTO',
        'i': word,
        'doctype': 'json',
        'version': '2.1',
        'keyfrom': 'fanyi.web',
        'ue': 'UTF-8',
        'action': 'FY_BY_CLICKBUTTON',
        'typeResult': 'true'
    }
    response = requests.post(url, data=key)

    if response.status_code == 200:
        return response.text
    else:
        print('有道词典调用失败')
        return None
    
def get_result(response):
    result = json.loads(response)
    # result: dict_keys(['type', 'errorCode', 'elapsedTime', 'translateResult'])
    # {"type":"ZH_CN2EN","errorCode":0,"elapsedTime":7,"translateResult":[[{"src":"车站里有人?","tgt":"Some people in the station?"}]]}
    trans_type = result['type']
    src_text = result['translateResult'][0][0]['src']
    tgt_text = result['translateResult'][0][0]['tgt']
    print('输入的词为:', src_text)
    print('翻译结果为:', tgt_text)
    return tgt_text, trans_type

def process_translate(sentence):
    list_trans = translate(sentence)
    result = json.loads(list_trans)
    # result: dict_keys(['type', 'errorCode', 'elapsedTime', 'translateResult'])
    # {"type":"ZH_CN2EN","errorCode":0,"elapsedTime":7,"translateResult":[[{"src":"车站里有人?","tgt":"Some people in the station?"}]]}
    trans_type = result['type']
    src_text = result['translateResult'][0][0]['src']
    tgt_text = result['translateResult'][0][0]['tgt']  
    return tgt_text, trans_type


def main():
    print('有道词典API翻译')
    text = input('输入你想要翻译的词或句：')
    list_trans = translate(text)
    get_result(list_trans)


if __name__ == '__main__':
    main()