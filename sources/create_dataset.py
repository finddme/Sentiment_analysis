import torch
from torch.utils.data import Dataset
from sources.mongo_processor import Mongo
import config

class Create_Dataset(Dataset):
    def __init__(self, datas, labels, tokenizer, max_len):
      self.datas = datas
      self.tokenizer = tokenizer
      self.max_len = max_len
      self.labels = labels
    
    def __len__(self):
      # from torch.utils.data import DataLoader -> len()을 필요로 함
      return len(self.datas)

    def __getitem__(self, index):
        #print(self.datas[index], index)
        inputs = self.tokenizer.encode_plus(text=self.datas[index],
                                           text_pair=None,
                                           add_special_tokens = True,
                                           max_length=self.max_len,
                                           padding='max_length',
                                           return_token_type_ids=False,
                                           return_attention_mask=True,
                                           truncation=True,
                                           return_tensors='pt' # PyTorch Tensor format
                                           )
        input_ids = inputs['input_ids'].flatten()
        attn_mask = inputs['attention_mask'].flatten()
        dataset = {'sentences': self.datas[index],
                    'input_ids': input_ids,
                    'attention_mask': attn_mask,
                    'labels': torch.tensor(self.labels[index], dtype=torch.float32)}


        return dataset


class Create_API_Data(Dataset):
    def __init__(self, datas, tokenizer, max_len):
      self.datas = datas
      self.tokenizer = tokenizer
      self.max_len = max_len
    
    def __len__(self):
      return len(self.datas)

    def __getitem__(self, index):
        inputs = self.tokenizer.encode_plus(text=self.datas[index],
                                           text_pair=None,
                                           add_special_tokens = True,
                                           max_length=self.max_len,
                                           padding='max_length',
                                           return_token_type_ids=False,
                                           return_attention_mask=True,
                                           truncation=True,
                                           return_tensors='pt' # PyTorch Tensor format
                                           )

        input_ids = inputs['input_ids'].flatten()
        attn_mask = inputs['attention_mask'].flatten()
        dataset = {'sentences': self.datas[index],
                    'input_ids': input_ids,
                    'attention_mask': attn_mask}


        return dataset

