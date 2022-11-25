from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
model = SentenceTransformer('bert-base-uncased')

article = pd.read_csv(r'data/article.csv',low_memory=False)
article.dropna(subset=['AB'],inplace=True)
article = article[article.DT.isin(['Article; Proceedings Paper','Article','Proceeding Paper'])]
article.drop_duplicates(inplace=True)

def article_sp(i):
    s = str(i).split('. ')
    return s
sp_article = list(article['AB'].apply(article_sp))
article_sentence_ = []
for i in range(0,len(sp_article)):
    #print(i)
    #print(i,sp_paper[i])
    for j in sp_article[i]:
        if '(C)' not in j and '(c)' not in j and 'All rights reserved.' not in j:
            article_sentence_.append(j)
article_sentence = [i for i in article_sentence_ if i !='']

patent = pd.read_csv('data/patent_sentence.csv')
patent.dropna(subset=['s'],inplace=True)
patent = patent.reset_index()
patent_sentence = patent['s']

#Compute embeddings
embeddings1 = model.encode(article_sentence,batch_size=32,convert_to_tensor=True,device='cpu')
torch.save(embeddings1,'data/embeddings_article.pt')
embeddings2 = model.encode(patent_sentence,batch_size=32, convert_to_tensor=True,device='cpu')
torch.save(embeddings2,'data/embeddings_patent.pt')