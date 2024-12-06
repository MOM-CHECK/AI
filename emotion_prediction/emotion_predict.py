import torch
from torch import nn
import numpy as np
from transformers import BertModel
from kobert_tokenizer import KoBERTTokenizer
from torch.utils.data import Dataset, DataLoader

max_len = 64
batch_size = 32
warmup_ratio = 0.1
num_epochs = 5
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

device = torch.device("cpu")

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len, pad, pair):
        self.sentences = []
        self.labels = []

        for i in dataset:
            # 문자열 변환 및 토큰화
            encoding = bert_tokenizer(
                str(i[sent_idx]),  # 입력 문장
                max_length=max_len,  # 최대 길이 설정
                padding='max_length',  # 패딩
                truncation=True,  # 길이 초과 시 자르기
                return_tensors='pt'  # PyTorch 텐서로 반환
            )

            # token_ids, attention_mask, segment_ids 추출
            self.sentences.append((
                encoding['input_ids'].squeeze(0),  # 배치 차원 제거
                encoding['attention_mask'].squeeze(0),
                encoding['token_type_ids'].squeeze(0)
            ))
            self.labels.append(np.int32(i[label_idx]))

    def __getitem__(self, i):
        return self.sentences[i] + (self.labels[i],)

    def __len__(self):
        return len(self.labels)


class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=4,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = valid_length  # valid_length는 이미 attention mask

        _, pooler = self.bert(
            input_ids=token_ids,
            token_type_ids=segment_ids.long(),
            attention_mask=attention_mask.float().to(token_ids.device),
            return_dict=False
        )

        if self.dr_rate:
            out = self.dropout(pooler)
        else:
            out = pooler

        return self.classifier(out)

def predict(predict_sentence):  # input = 감정분류하고자 하는 sentence
    device = torch.device("cpu")
    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
    bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
    model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)

    model_path = './models/trained_kobert_model.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 반드시 evaluation 모드로 전환

    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tokenizer, max_len, True, False)  # 토큰화한 문장
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)  # torch 형식 변환

    model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length = valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)

        # out이 배치가 하나만 있는 경우, 차원을 축소
        if out.ndimension() == 2 and out.shape[0] == 1:  # 배치가 하나만 있는 경우
            logits = out.squeeze(0).detach().cpu().numpy()
        elif out.ndimension() == 1:  # 이미 배치가 없는 경우
            logits = out.detach().cpu().numpy()
        else:
            raise ValueError("Unexpected output shape from the model")

        # np.argmax 호출 전에 logits 차원 확인
        if logits.ndim == 1:
            emotion_labels = ["슬픔", "불안", "분노", "기쁨"]
            predicted_emotion = emotion_labels[np.argmax(logits)]
            return predicted_emotion
        else:
            print("Unexpected output shape, logits:", logits.shape)
            return None
