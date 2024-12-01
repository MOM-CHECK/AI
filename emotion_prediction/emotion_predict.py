import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from transformers import BertModel

from emotion_prediction.train_data_processing import BERTDataset

max_len = 64
batch_size = 32
device = torch.device("cpu")

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes = 4,   # 감정 클래스 수 수정
                 dr_rate = None,
                 params = None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p = dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device),return_dict = False)
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

def make_BERT_model():
    bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
    model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
    return model

def load_model():
    model = make_BERT_model()
    model_path = './models/trained_kobert_model.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 반드시 evaluation 모드로 전환

def get_emotion(emotion_number):
    if emotion_number == 0:
        test_eval.append("슬픔")
    elif emotion_number == 1:
        test_eval.append("불안")
    elif emotion_number == 2:
        test_eval.append("분노")
    elif emotion_number == 3:
        test_eval.append("기쁨")

def predict(predict_sentence):
    model = load_model()

    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tokenizer, vocab, max_len, True, False) # 토큰화한 문장
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size = batch_size, num_workers = 5) # torch 형식 변환

    model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length = valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)

        test_eval = []
        for i in out: # out = model(token_ids, valid_length, segment_ids)
            logits = i
            logits = logits.detach().cpu().numpy()
            get_emotion(np.argmax(logits))

        return test_eval[0]
