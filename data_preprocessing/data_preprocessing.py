import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

from kobert_tokenizer import KoBERTTokenizer

import gluonnlp as nlp

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, vocab, max_len, pad, pair):

        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len,vocab = vocab, pad = pad, pair = pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))


    def __len__(self):
        return (len(self.labels))

def emotion_to_number(data):
    data.loc[(data['emotion'] == "Sadness"), 'emotion'] = 0  # Sadness → 0
    data.loc[(data['emotion'] == "Anxiety"), 'emotion'] = 1  # Anxiety → 1
    data.loc[(data['emotion'] == "Anger"), 'emotion'] = 2  # Anger → 2
    data.loc[(data['emotion'] == "Joy"), 'emotion'] = 3  # Joy → 3

def map_emotions(emotion):
    if emotion in ['Panic', 'Wound', 'Sadness']:
        return 'Sadness'
    return emotion

def make_list(data):
    data_list = []
    for ques, label in zip(data['talk'], data['emotion']):
        data = []
        data.append(ques)
        data.append(str(label))
        data_list.append(data)
    return data_list

def data_processing():
    # 1. 두 학습 데이터 로드 및 병합
    train_data = pd.read_csv('../csv/청년여성_연애결혼출산_학습데이터.csv')
    test_data = pd.read_csv('../csv/청년_여성_결혼연애출산_테스트_데이터.csv')

    # 감정 통합 적용
    train_data['emotion'] = train_data['emotion'].apply(map_emotions)
    test_data['emotion'] = test_data['emotion'].apply(map_emotions)

    # 4개의 감정 class → 숫자
    emotion_to_number(train_data)
    emotion_to_number(test_data)

    train_data_list = make_list(train_data)
    test_data_list = make_list(test_data)

    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
    vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')

    # 파라미터 세팅
    max_len = 64
    batch_size = 32

    # BERTDataset : 각 데이터가 BERT 모델의 입력으로 들어갈 수 있도록 tokenization, int encoding, padding하는 함수
    data_train = BERTDataset(train_data_list, 0, 1, tokenizer, vocab, max_len, True, False)
    data_test = BERTDataset(test_data_list, 0, 1, tokenizer, vocab, max_len, True, False)

    # torch 형식의 dataset을 만들어 입력 데이터셋의 전처리 마무리
    train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=5)
    test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=5)

    return train_dataloader, test_dataloader
