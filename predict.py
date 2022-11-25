import torch.nn as nn
from normal_bert import ClassificationBert
import torch.utils.data as Data
import torch
from read_data import loader_unlabeled
from pytorch_transformers import *
import numpy as np
import pandas as pd

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_seq_len = 256

test_df = pd.read_csv(r'data\patent_sentence.csv', header=None)
#labels = np.array([v-1 for v in test_df[1]])
text = np.array([v for v in test_df[1]])
train_unlabeled_dataset = loader_unlabeled(text, tokenizer, max_seq_len)
data_loader = Data.DataLoader(dataset=train_unlabeled_dataset, batch_size=20, shuffle=False)
n_labels = 5
model = ClassificationBert(n_labels)

model = nn.DataParallel(model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model = model.to(device)
model.load_state_dict(torch.load(r'ns.pt'))
model.eval()

category = []

for batch_idx, (inputs, length) in enumerate(data_loader):
    input = inputs
    input = input.cpu()
    output = model(input)
    _, predicted = torch.max(output.data, 1)
    #print(np.array(predicted.cpu()))
    category += list(np.array(predicted.cpu()))
category = pd.DataFrame({'label':category})
df = test_df.join(category)
df.to_csv(r"data\patent_sentence_label.csv",index=False,header=['patent_number','sentence','label'])
