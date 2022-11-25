from sentence_transformers import util
import torch
import json

f1 = 'data/embeddings_article.pt'
f2 = 'data/embeddings_patent.pt'
article_embeddings = torch.load(f1, map_location=torch.device('cpu'))
patent_embeddings = torch.load(f2, map_location=torch.device('cpu'))
print(len(article_embeddings))
print(len(patent_embeddings))

def sim_compute(embeddings1, embeddings2):
    # Compute cosine-similarities for each sentence with each other sentence
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

    score = []
    for i in range(len(cosine_scores)):
        for j in range(i + 1, len(cosine_scores[0])):
            score.append(float(cosine_scores[i][j]))

    return score

len_article = len(article_embeddings)
print(len_article)
for i in range(0,len_article,100):
    print(i,i+100)
    if i/100 != 12727:
        embeddings1 = article_embeddings[i:i+100]
    else:
        embeddings1 = article_embeddings[1272700:len_article]
    score = sim_compute(embeddings1,patent_embeddings)
    s = ['{:.4f}'.format(i) for i in score]
    dict = {}
    for key in s:
        dict[key] = dict.get(key, 0) + 1
    json_str = json.dumps(dict)
    with open('data/ap/score_num'+str(i/100)+'.json', 'w') as json_file:
        json_file.write(json_str)

len_patent = len(patent_embeddings)
print(len_patent)
for i in range(0,len_article,100):
    print(i,i+100)
    if i/100 != 12727:
        embeddings1 = patent_embeddings[i:i+100]
    else:
        embeddings1 = patent_embeddings[1272700:len_patent]
    score = sim_compute(embeddings1,patent_embeddings)
    s = ['{:.4f}'.format(i) for i in score]
    dict = {}
    for key in s:
        dict[key] = dict.get(key, 0) + 1
    json_str = json.dumps(dict)
    with open('data/pp/score_num'+str(i/100)+'.json', 'w') as json_file:
        json_file.write(json_str)
