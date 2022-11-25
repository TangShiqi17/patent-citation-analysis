import pandas as pd
from numpy import *

def calculate_contribution(file):
    patent = pd.read_csv("data/"+file)
    df = pd.DataFrame({})
    sim = []
    patent_number = []

    if file == 'patent_to_patent_sim.csv':
        for i in patent['similarity'].groupby(patent['patent_number_cite']):
            patent_number.append(i[0])
            sim.append(i[1].values)
    else:
        for i in patent['similarity'].groupby(patent['patent_number']):
            patent_number.append(i[0])
            sim.append(i[1].values)

    #print(contri_paper)
    sim = pd.DataFrame({'sim_patent':sim})
    df = df.append(sim)
    #print(df)
    contri = []
    l0 = []
    l1 = []
    l2 = []
    l3 = []
    l4 = []
    for i in df['sim']:
        contri0 = 0
        contri1 = 0
        contri2 = 0
        contri3 = 0
        contri4 = 0
        for e in i:
            for key in eval(e).keys():
                if len(eval(e)[key]) > 0:
                    for j in eval(e)[key]:
                        if j[0] == '0':
                            contri0 += j[1]
                            l0.append(j[1])
                        elif j[0] == '1':
                            contri1 += j[1]
                            l1.append(j[1])
                        elif j[0] == '2':
                            contri2 += j[1]
                            l2.append(j[1])
                        elif j[0] == '3':
                            contri3 += j[1]
                            l3.append(j[1])
                        elif j[0] == '4':
                            contri4 += j[1]
                            l4.append(j[1])
        contri.append({'0':contri0,'1':contri1,'2':contri2,'3':contri3,'4':contri4})

    patent_number = pd.DataFrame({'patent_number':patent_number})
    if file == 'paper_to_patent_sim.csv':
        contri = pd.DataFrame({'contri_paper':contri})
        df = pd.concat([patent_number,df,contri],axis=1)
        df.to_csv('data/contri_paper.csv',index=False)
    if file == 'patent_to_patent_sim.csv':
        contri = pd.DataFrame({'contri_patent':contri})
        df = pd.concat([patent_number,df,contri],axis=1)
        df.to_csv('data/contri_patent.csv',index=False)

calculate_contribution('paper_to_patent_sim.csv')
calculate_contribution('patent_to_patent_sim.csv')

contri_paper = pd.read_csv('data/contri_paper.csv')
contri_patent = pd.read_csv('data/contri_patent.csv')
contri = pd.merge(contri_patent,contri_paper,on='patent_number',how='left',validate='1:1')
#calculate the contribution degree of each knowledge category as con
#calculate the contribution degree of paper in each category as con_paper
#calculate the contribution degree of patent in each category as con_patent
con = []
con_paper = []
con_patent = []
for i in range(0,len(contri)):
    print(i)
    total = {'0':0,'1':0,'2':0,'3':0,'4':0}
    cond_paper = {'0':0,'1':0,'2':0,'3':0,'4':0}
    cond_patent = {'0':0,'1':0,'2':0,'3':0,'4':0}
    if str(contri['contri_paper'][i]) != 'nan':
        paper = eval(str(contri['contri_paper'][i]))
        for key in paper.keys():
            total[key] += paper[key]
    if str(contri['contri_patent'][i]) != 'nan':
        patent = eval(str(contri['contri_patent'][i]))
        for key in patent.keys():
            total[key] += patent[key]
    cond = {'0':0,'1':0,'2':0,'3':0,'4':0}
    sum = 0
    for key in total.keys():
        sum += total[key]
    if sum != 0:
        for key in total.keys():
            cond[key] = total[key]/sum
            if str(contri['contri_paper'][i]) != 'nan' and total[key] != 0:
                paper = eval(str(contri['contri_paper'][i]))
                cond_paper[key] = (paper[key] / total[key])*cond[key]
            if str(contri['contri_patent'][i]) != 'nan' and total[key] != 0:
                patent = eval(str(contri['contri_patent'][i]))
                cond_patent[key] = (patent[key] / total[key])*cond[key]
    con.append(cond)
    con_paper.append(cond_paper)
    con_patent.append(cond_patent)
con = pd.DataFrame({'con':con})
con_paper = pd.DataFrame({'con_paper':con_paper})
con_patent = pd.DataFrame({'con_patent':con_patent})
contri = pd.concat([contri,con,con_paper,con_patent],axis=1)
contri.to_csv('data/con.csv',index=False)
