import json
import os
import pandas as pd
import matplotlib.pyplot as plt

#sort to get the top 20% appearance values as the threshold
def threshold(folder_path):
    file_list = os.listdir(folder_path)
    ap = {}
    for file in file_list:
        with open(folder_path + '/' + file,'r',encoding='utf8')as fp:
            json_data = json.load(fp)
            for sim in list(json_data):
                if sim not in ap.keys():
                    ap[sim] = json_data[sim]
                else:
                    ap[sim] += json_data[sim]

    ap_total = sum(ap.values())
    print(ap_total)
    ap_rank = sorted(ap.items(), key = lambda ap:[ap[0], ap[1]],reverse=True)
    df = pd.DataFrame(ap_rank)
    #print(ap_rank)
    num = 0
    flag = 0
    for i in ap_rank:
        if flag == 0:
            num_1 = i[1]
            print(num_1)
        num += i[1]
        flag += 1
        if num >= ap_total*0.2+num_1:
            print(i[0])
            break
    return df

folder_path_ap = 'data/ap'
df_ap = threshold(folder_path_ap)
folder_path_pp = 'data/pp'
df_pp = threshold(folder_path_pp)
df_ap.columns = ['similarity', 'count']
df_pp.columns = ['similarity', 'count']
max_ap_index = df_ap['count'].argmax()
max_pp_index = df_pp['count'].argmax()
max_ap = df_ap['similarity'][max_ap_index]
max_pp = df_pp['similarity'][max_pp_index]
print(max_ap,max_pp)
plt.plot(df_pp['similarity'],df_pp['count'],color='r',label='patent-patent')
plt.plot(df_ap['similarity'],df_ap['count'],label='paper-patent')
"""plt.axvline(0.7921, color='r', linestyle='--')
plt.axvline(0.7725, linestyle='--')"""
plt.xticks([])
plt.legend(fontsize=15)
plt.xlabel('similarity', fontsize=15)
plt.ylabel('count', fontsize=15)
plt.show()