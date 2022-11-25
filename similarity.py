from sentence_transformers import SentenceTransformer, util
from ast import literal_eval
import pandas as pd

def calculate_similarity(file,threshold):
    df = pd.read_csv("data/"+file)
    model = SentenceTransformer('bert-base-uncased')

    similarity = []
    # Single list of sentences
    for i in range(0,len(df)):
        print(i)
        if file == 'paper_to_patent.csv':
            sentences1 = literal_eval(df['paper_sentence_list'][i])
            sentences2 = literal_eval(df['patent_sentence_list'][i])
        if file == 'patent_to_patent.csv':
            sentences1 = literal_eval(df['patent_sentence_list_cited'][i])
            sentences2 = literal_eval(df['patent_sentence_list_cite'][i])

        #Compute embeddings
        embeddings1 = model.encode(sentences1, convert_to_tensor=True)
        embeddings2 = model.encode(sentences2, convert_to_tensor=True)

        #Compute cosine-similarities for each sentence with each other sentence
        cosine_scores = util.pytorch_cos_sim(embeddings2, embeddings1)

        #Find the pairs with the highest cosine similarity scores
        #保留相似度大于阈值的语句对
        #paper_to_patent 0.7725
        #patent_to_patent 0.7921
        sim = {}
        for j in range(0,len(cosine_scores)): #j patent cite
            cite = []
            for k in range(0, len(cosine_scores[0])): #k paper cited
                #print('{} {}'.format(j,k))
                if cosine_scores[j][k] > threshold:
                    s = cosine_scores[j][k]
                    cite.append((str(df['label_list'][i]).replace('[','').replace(']','').split(' ')[k],float(s)))
            sim[j] = cite
        #print(sim)

        #Sort scores in decreasing order
        #pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)

        similarity.append(sim)

    similarity = pd.DataFrame({'similarity':similarity})
    data = df.join(similarity)
    data.to_csv("data/"+file,index=False)

calculate_similarity('paper_to_patent.csv',0.7725)
calculate_similarity('patent_to_patent.csv',0.7921)
