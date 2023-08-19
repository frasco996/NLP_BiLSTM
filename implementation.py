import numpy as np
from typing import List
import torch.nn as nn
import gensim
from model import Model
import torch
import json
import torch.nn.functional as F

def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    return StudentModel(device)
    #return RandomBaseline()
    

class RandomBaseline(Model):
    options = [
        (22458, "B-ACTION"),
        (13256, "B-CHANGE"),
        (2711, "B-POSSESSION"),
        (6405, "B-SCENARIO"),
        (3024, "B-SENTIMENT"),
        (457, "I-ACTION"),
        (583, "I-CHANGE"),
        (30, "I-POSSESSION"),
        (505, "I-SCENARIO"),
        (24, "I-SENTIMENT"),
        (463402, "O")
    ]

    def __init__(self):
        self._options = [option[1] for option in self.options]
        self._weights = np.array([option[0] for option in self.options])
        self._weights = self._weights / self._weights.sum()

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        return [
            [str(np.random.choice(self._options, 1, p=self._weights)[0]) for _x in x]
            for x in tokens
        ]
#CREATE BILSTM MODULE
class BiLSTM(nn.Module):
  def __init__(self, embedding_dim, hidden_dim, num_layer, dropout,num_classes,w2v):
    super(BiLSTM, self).__init__()
    self.embedding_dim = embedding_dim
    self.hidden_dim = hidden_dim
    self.num_layers = num_layer
    self.dropout = dropout

    self.embedding= nn.Embedding.from_pretrained(embeddings=torch.FloatTensor(w2v.vectors),padding_idx=0,freeze=True)
    self.emb_dropout = nn.Dropout(dropout)

    self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, 
                            num_layers=num_layer,
                            bidirectional= True,
                            dropout = dropout)

    self.c_dropout = nn.Dropout(dropout)
    self.classifier = nn.Linear(hidden_dim * 2, num_classes)
  
  def forward(self, x):
    embedding_drop=self.emb_dropout(self.embedding(x))
    lstm, _ = self.lstm(embedding_drop.view(len(x),1,-1))
    classi = self.classifier(self.c_dropout(lstm.view(len(x),-1)))
    softmax = F.log_softmax(classi,dim=1)
    return softmax
    
#CREATE DATASET    
class Dataset():

    #READ FILE
    def datat(self,input):
        f = open (input, "r")
        data = json.loads(f.read())
        f.close()
        return data
        
    #LIST OF IDX        
    def val_lists(self,data,vocab):
        tokens_idx = []
        for i in range(0,len(data)):
            token = []
            for j in range(0,len(data[i])):
                if(data[i][j] in vocab):
                    token.append(vocab[data[i][j]])
                else:
                    token.append(1)
            tokens_idx.append(token)
        return tokens_idx        
  
    
class StudentModel(Model):

    def __init__(self,device):
        super(StudentModel, self).__init__()
        self.device = device
        self.embedding_dim = 300
        self.hidden_dim = 350
        self.num_layer = 2
        self.dropout = 0.25
        self.num_classes = 11
        self.dataset=Dataset()
        self.vocab = self.dataset.datat('model/vocab.json') #IMPORT VOCABULARY
        self.w2v = gensim.models.KeyedVectors.load_word2vec_format('model/word2vec-google-news-300', binary=True)  #IMPORT WORD2VEC
        self.labels = {'O':0 , 'B-SCENARIO': 1  , 'B-SENTIMENT': 2, 'B-CHANGE':3, 'B-ACTION': 4, 'B-POSSESSION': 5 ,'I-SCENARIO': 6 , 'I-SENTIMENT': 7, 'I-CHANGE': 8, 'I-ACTION': 9,'I-POSSESSION' :10 }
        self.model = BiLSTM(self.embedding_dim, self.hidden_dim, self.num_layer, self.dropout,self.num_classes,self.w2v).to(self.device)
        self.model.load_state_dict(torch.load('model/state7027.pt', map_location=torch.device(self.device)))
        
    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        tokens_idx = self.dataset.val_lists(tokens,self.vocab)
        self.model.eval()
        labels_idx = []
        labels_f = []
        for i in range(len(tokens_idx)):
            sentence = tokens_idx[i]
            tensor_tokens = torch.tensor(sentence, dtype=torch.long).to(self.device)
            predicted = self.model(tensor_tokens)
            _,indices = torch.max(predicted,1)
            labels_idx.append(indices)
        #TRANSFORM PREDICTED LABELS IN STRING
        for i in labels_idx:
            label = []
            for j in i:
                for key, value in self.labels.items():
                    if j == value:
                        label.append(key)
            labels_f.append(label)
        return labels_f
