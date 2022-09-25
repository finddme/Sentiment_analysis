import torch
import torch.nn as nn
from transformers import ElectraModel, ElectraTokenizer,  ElectraConfig
from transformers import BertModel
import config
from sources.parallel import DataParallelModel, DataParallelCriterion

class KoElectra(torch.nn.Module):
    def __init__(self):
        super(KoElectra, self).__init__()
        self.num_labels = 34
        self.model = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
        self.dropout = torch.nn.Dropout(0.1)
        self.linear = torch.nn.Linear(self.model.config.hidden_size, self.num_labels)
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        # self.loss = torch.nn.Sigmoid()
    def forward(self, input_ids, labels):

        discriminator_hidden_states = self.model(input_ids)
        
        pooled_output = discriminator_hidden_states[0][:, 0] #(64, 768)
        pooled_output = self.dropout(pooled_output) #(64, 768)
        logits = self.linear(pooled_output) #(64, 34)
        outputs = (logits,) + discriminator_hidden_states[1:]

        softmax = torch.nn.functional.softmax(outputs[0], dim=1) #(64, 34)
        #real = torch.nn.functional.softmax(labels)

        pred = softmax.argmax(dim=1) #(64)
        #real = real.argmax()
        #pred = pred.tolist()[0]
        #real = real.tolist()[0]

        #correct = pred == real.item()
        correct = pred.eq(labels)  #(64)

        # logits = self.sigmoid(logits)
        loss = self.loss_fct(logits, labels)
        
        outputs = (loss,) + outputs # loss, logits, hidden_states

        return outputs, pred, labels, correct, softmax

class KoElectra_m(torch.nn.Module):
    def __init__(self):
        super(KoElectra_m, self).__init__()
        self.num_labels = 34
        self.model = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
        self.dropout = torch.nn.Dropout(0.1)
        self.linear = torch.nn.Linear(self.model.config.hidden_size, self.num_labels)
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        # self.loss = torch.nn.Sigmoid()
    def forward(self, input_ids, labels, device):

        discriminator_hidden_states = self.model(input_ids)
        pooled_output = discriminator_hidden_states[0][:, 0] #(64, 768)
        pooled_output = self.dropout(pooled_output) #(64, 768)
        logits = self.linear(pooled_output) #(64, 34)
        softmax = torch.nn.functional.softmax(logits, dim=1) #(64, 34)
        #real = torch.nn.functional.softmax(labels)
    
        pred = softmax.argmax(dim=1) #(64)

        correct = pred.eq(labels)  #(64)

        # criterion = DataParallelCriterion(self.loss_fct)
        criterion = self.loss_fct.to(device)
        loss = criterion(logits, labels)

        # loss = self.loss_fct(logits, labels)
        
        return loss, pred, labels, correct, softmax

class KoElectra_api(torch.nn.Module):
    def __init__(self):
        super(KoElectra_api, self).__init__()
        self.num_labels = 34
        self.model = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
        self.dropout = torch.nn.Dropout(0.1)
        self.linear = torch.nn.Linear(self.model.config.hidden_size, self.num_labels)
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        # self.loss = torch.nn.Sigmoid()
    def forward(self, input_ids):

        discriminator_hidden_states = self.model(input_ids)
        
        pooled_output = discriminator_hidden_states[0][:, 0] #(64, 768)
        pooled_output = self.dropout(pooled_output) #(64, 768)
        logits = self.linear(pooled_output) #(64, 34)
        outputs = (logits,) + discriminator_hidden_states[1:]

        softmax = torch.nn.functional.softmax(outputs[0], dim=1) #(64, 34)

        pred = softmax.argmax(dim=1) #(64)


        return pred, softmax

class Kobert(torch.nn.Module):
    def __init__(self):
        super(Kobert, self).__init__()
        self.num_labels = 34
        self.model = BertModel.from_pretrained("monologg/kobert")
        self.dropout = torch.nn.Dropout(0.1)
        self.linear = torch.nn.Linear(self.model.config.hidden_size, self.num_labels)
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        # self.loss = torch.nn.Sigmoid()
    def forward(self, input_ids, labels):

        discriminator_hidden_states = self.model(input_ids)
        
        pooled_output = discriminator_hidden_states[1] #(64, 768)
        pooled_output = self.dropout(pooled_output) #(64, 768)
        logits = self.linear(pooled_output) #(64, 34)
        outputs = (logits,) + discriminator_hidden_states[2:]

        softmax = torch.nn.functional.softmax(outputs[0], dim=1) #(64, 34)
        #real = torch.nn.functional.softmax(labels)

        pred = softmax.argmax(dim=1) #(64)
        #real = real.argmax()
        #pred = pred.tolist()[0]
        #real = real.tolist()[0]

        #correct = pred == real.item()
        correct = pred.eq(labels)  #(64)

        # logits = self.sigmoid(logits)
        loss = self.loss_fct(logits, labels)
        
        outputs = (loss,) + outputs # loss, logits, hidden_states

        return outputs, pred, labels, correct, softmax

class Kobert_m(torch.nn.Module):
    def __init__(self):
        super(Kobert_m, self).__init__()
        self.num_labels = 34
        self.model = BertModel.from_pretrained("monologg/kobert")
        self.dropout = torch.nn.Dropout(0.1)
        self.linear = torch.nn.Linear(self.model.config.hidden_size, self.num_labels)
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        # self.loss = torch.nn.Sigmoid()
    def forward(self, input_ids, labels, device):

        discriminator_hidden_states = self.model(input_ids)
        pooled_output = discriminator_hidden_states[1] #(64, 768)
        pooled_output = self.dropout(pooled_output) #(64, 768)
        logits = self.linear(pooled_output) #(64, 34)
        softmax = torch.nn.functional.softmax(logits, dim=1) #(64, 34)
        #real = torch.nn.functional.softmax(labels)
    
        pred = softmax.argmax(dim=1) #(64)

        correct = pred.eq(labels)  #(64)

        # criterion = DataParallelCriterion(self.loss_fct)
        criterion = self.loss_fct.to(device)
        loss = criterion(logits, labels)

        # loss = self.loss_fct(logits, labels)
        
        return loss, pred, labels, correct, softmax