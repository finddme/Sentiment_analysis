from flask import request, make_response
from flask_restful import Resource
from sources.model import KoElectra_api
from transformers import ElectraTokenizer
from sources.create_dataset import Create_API_Data
from pytorch_transformers import AdamW
from torch.utils.data import DataLoader
import torch
import json
import logging
from sources.utils import init_logger
import config
import copy

init_logger()
logger = logging.getLogger(__name__)

class API(Resource): 
    def get(self):
        global args
        global model
        global tokenizer
        global device

        result_dict = {}
        inputs = request.args.get('sentence', None)

        if inputs is None or len(inputs) < 1:
            result_dict = dict(message="fail")
            return json.dumps(result_dict, ensure_ascii=False)

        encodings = tokenizer.encode_plus(inputs,
                                        None,
                                        add_special_tokens=True, 
                                        max_length=128,
                                        padding='max_length',
                                        return_token_type_ids=False,
                                        return_attention_mask=True, 
                                        truncation=True,  
                                        return_tensors='pt')
                        

        with torch.no_grad():
            ids = encodings['input_ids'].to(device, dtype=torch.long)
            mask = encodings['attention_mask'].to(device, dtype=torch.long)
            pred,softmax = model(ids)
        
        softmax = softmax.tolist()[0]

        emotion_7 ={'중립':0, '싫음':0, '행복':0, '슬픔':0, '두려움':0, '분노':0, '놀람':0,'사랑':0}
        emotion_3 = {'중립':0,'긍정':0,'부정':0}
        emotion_34 = {}

        for index, result in enumerate(softmax):
            emotion_34[index] = result
            if index in config.emotion7_0:
                emotion_7['중립'] += result 
            elif index in config.emotion7_1:
                emotion_7['싫음'] += result
            elif index in config.emotion7_2:
                emotion_7['행복'] += result
            elif index in config.emotion7_3:
                emotion_7['슬픔'] += result
            elif index in config.emotion7_4:
                emotion_7['두려움'] += result
            elif index in config.emotion7_5:
                emotion_7['분노'] += result
            elif index in config.emotion7_6:
                emotion_7['사랑'] += result
            elif index in config.emotion7_7:
                emotion_7['놀람'] += result

            if index in config.pos_emotion:
                emotion_3['긍정'] += result
            elif index in config.neg_emotion:
                emotion_3['부정'] += result
            else:
                emotion_3['중립'] += result
        
        emotion_34 = dict([(config.e_34.get(key), value) for key, value in emotion_34.items()])
        emotion_34 = dict(sorted(emotion_34.items(), key=lambda x : x[1], reverse=True))
        emotion_7 = dict(sorted(emotion_7.items(), key=lambda x : x[1], reverse=True))
        emotion_3 = dict(sorted(emotion_3.items(), key=lambda x : x[1], reverse=True))

        
        
        max_values_7 = max(emotion_7.values())
        max_key_7 = {v:k for k,v in emotion_7.items()}.get(max_values_7)
        max_values_3 = max(emotion_3.values())
        max_key_3 = {v:k for k,v in emotion_3.items()}.get(max_values_3)
        max_values_34 = max(emotion_34.values())
        max_key_34 = {v:k for k,v in emotion_34.items()}.get(max_values_34)

        result_dict['sentence'] = inputs
        result_dict['result(emotion_8)'] = max_key_7
        result_dict['result(emotion_34)'] = max_key_34
        result_dict['result(emotion_3)'] = max_key_3
        result_dict['emotion_8'] = emotion_7
        result_dict['emotion_34'] = emotion_34
        result_dict['emotion_3'] = emotion_3
        result_dict['label_info'] = config.label_info

        #jsonify(result_dict)
        return make_response(json.dumps(result_dict,  ensure_ascii=False))

    def post(self):
        global args
        global model
        global tokenizer
        global device

        result_dict = {}
        inputs = request.get_json().get('sentences', None)
        result_dict['sentences'] = copy.deepcopy(inputs)

        if inputs is None or len(inputs) < 1:
            result_dict = dict(message="fail")
            return json.dumps(result_dict, ensure_ascii=False)
            
        encodings = Create_API_Data(datas=inputs,
                                    max_len=config.MODEL_CONFIG['max_len'],
                                    tokenizer=tokenizer)
        encodings = DataLoader(encodings, batch_size=config.MODEL_CONFIG['batch_size'], shuffle=False, num_workers=0)
                        

        with torch.no_grad():
            emotion_7_total =[]
            emotion_34_total = []
            emotion_3_total =[]
            max_key_7_total =[]
            max_key_3_total =[]
            max_key_34_total =[]
            for batch_idx, data in enumerate(encodings):
                ids = data['input_ids'].to(device, dtype=torch.long)
                mask = data['attention_mask'].to(device, dtype=torch.long)
                pred,softmax = model(ids)
                softmax = softmax.tolist()

                for i in softmax:
                    emotion_7 ={'중립':0, '싫음':0, '행복':0, '슬픔':0, '두려움':0, '분노':0, '놀람':0, '사랑':0}
                    emotion_3 = {'중립':0,'긍정':0,'부정':0}
                    emotion_34 = {}
                    for index, result in enumerate(i):
                        emotion_34[index] = result
                        if index in config.emotion7_0:
                            emotion_7['중립'] += result 
                        elif index in config.emotion7_1:
                            emotion_7['싫음'] += result
                        elif index in config.emotion7_2:
                            emotion_7['행복'] += result
                        elif index in config.emotion7_3:
                            emotion_7['슬픔'] += result
                        elif index in config.emotion7_4:
                            emotion_7['두려움'] += result
                        elif index in config.emotion7_5:
                            emotion_7['분노'] += result
                        elif index in config.emotion7_6:
                            emotion_7['사랑'] += result
                        elif index in config.emotion7_7:
                            emotion_7['놀람'] += result  

                        if index in config.pos_emotion:
                            emotion_3['긍정'] += result
                        elif index in config.neg_emotion:
                            emotion_3['부정'] += result
                        else:
                            emotion_3['중립'] += result
                    
                    emotion_34 = dict([(config.e_34.get(key), value) for key, value in emotion_34.items()])
                    emotion_34 = dict(sorted(emotion_34.items(), key=lambda x : x[1], reverse=True))
                    emotion_7 = dict(sorted(emotion_7.items(), key=lambda x : x[1], reverse=True))
                    emotion_3 = dict(sorted(emotion_3.items(), key=lambda x : x[1], reverse=True))

                    emotion_7_total.append(emotion_7)
                    emotion_34_total.append(emotion_34)

                    max_values_7 = max(emotion_7.values())
                    max_key_7 = {v:k for k,v in emotion_7.items()}.get(max_values_7)
                    max_values_3 = max(emotion_3.values())
                    max_key_3 = {v:k for k,v in emotion_3.items()}.get(max_values_3)
                    max_values_34 = max(emotion_34.values())
                    max_key_34 = {v:k for k,v in emotion_34.items()}.get(max_values_34)

                    max_key_7_total.append(max_key_7)
                    max_key_3_total.append(max_key_3)
                    max_key_34_total.append(max_key_34)

        result_dict['result(emotion_8)'] = max_key_7_total
        result_dict['result(emotion_34)'] = max_key_34_total
        result_dict['result(emotion_3)'] = max_key_3_total
        result_dict['emotion_8'] = emotion_7_total
        result_dict['emotion_34'] = emotion_34_total
        result_dict['emotion_3'] = emotion_3_total
        result_dict['label_info'] = config.label_info
        #jsonify(result_dict)

        return make_response(json.dumps(result_dict,  ensure_ascii=False))

def load_model():
    global args
    global model
    global tokenizer
    global device

    if torch.cuda.is_available():
        device = torch.device('cuda:' + str(args.target_gpu))
        logger.info('There are %d GPU(s) available.' % torch.cuda.device_count())
        logger.info('We will use the GPU:{}'.format(torch.cuda.get_device_name(0)))
    else:
        device = torch.device("cpu")
        logger.info('No GPU available, using the CPU instead.')

    model = KoElectra_api()
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters,
                        lr=config.MODEL_CONFIG['learning_rate'],
                        eps=config.MODEL_CONFIG['adam_epsilon'])

    model.load_state_dict(torch.load(args.load_ck))
    model.eval()

    tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
