import json
import networkx as nx
import os
import pandas as pd
import numpy as np
from karateclub import Graph2Vec
from scipy import spatial
import matplotlib.pyplot as plt


def graph_embedding(A,B):
    graphs = [A,B]
    model = Graph2Vec()
    model.fit(graphs)
    X = model.get_embedding()
    dist_1 = 1 - spatial.distance.cosine(X[0], X[1])

    return dist_1


temp_path = 'C:/Users/seong/Desktop/app_front/graph_result/'



def graph_sim():
    in_path = os.listdir(temp_path)
    with open(temp_path+in_path[0], 'r') as r_json_file:
        temp_data = json.load(r_json_file)

    g1 = nx.node_link_graph(temp_data)

    with open(temp_path+in_path[1], 'r') as r_json_file:
        temp_data2 = json.load(r_json_file)

    g2 = nx.node_link_graph(temp_data2)



    for v in nx.optimize_graph_edit_distance(g1, g2):
        minv = v
        break
    print()
    print(f'Graph Similarity using GED = {minv}')
    print(graph_embedding(g1, g2))
    return minv




def generalization(target, database):
    tt = (target - np.min(database)) / (np.max(database)-np.min(database))

    return tt



def graph_calculation(fname1, fname2):
    with open(fname1, 'r') as r_json_file:
        temp_data = json.load(r_json_file)

    g1 = nx.node_link_graph(temp_data)

    with open(fname2, 'r') as r_json_file:
        temp_data2 = json.load(r_json_file)

    g2 = nx.node_link_graph(temp_data2)

    for v in nx.optimize_graph_edit_distance(g1, g2):
        minv = v
        break
    print()
    print(f'Graph Similarity using GED = {minv}')


    data = pd.read_csv('./result/result_final.csv')
    data_sample1 = data['Topology']

    s1 = pd.DataFrame([minv])

    data_sample1 = pd.concat([data_sample1, s1])

    cnt1 = 0

    d1 = []

    for i in range(len(data_sample1)):
        if pd.isna(data_sample1.iloc[i, 0]) == False:
            d1.append(data_sample1.iloc[i, 0])
        elif pd.isna(data_sample1.iloc[i, 0]) == True:
            cnt1 += 1

    d1 = sorted(d1)  ########### 클수록 좋으면 Reverse = True 추가

    r1 = d1.index(float(minv)) + cnt1 + 1

    f_score = 1 - generalization(minv, d1)  ############## 작을수록 유사도가 높기 때문에 1에서 빼준다.

    print(f'Grpah 순위 = {r1}/{len(d1)}')

    return f_score, f'{r1}/{len(d1)}'



def save_fig_plot(A,B):
    with open(A, 'r') as r_json_file:
        temp_data = json.load(r_json_file)
    g1 = nx.node_link_graph(temp_data)
    nx.draw(g1)
    plt.savefig('./temp_result/tt_i1.png', format="PNG")

    with open(B, 'r') as r_json_file:
        temp_data = json.load(r_json_file)
    g2 = nx.node_link_graph(temp_data)
    nx.draw(g2)
    plt.savefig('./temp_result/tt_i2.png', format="PNG")



if __name__ == '__main__':
    graph_score = graph_sim()
    data = pd.read_csv('./result/result_final.csv')
    data_sample1 = data['Topology']

    s1 = pd.DataFrame([graph_score])

    data_sample1 = pd.concat([data_sample1, s1])

    cnt1 = 0

    d1 = []

    for i in range(len(data_sample1)):
        if pd.isna(data_sample1.iloc[i, 0]) == False:
            d1.append(data_sample1.iloc[i, 0])
        elif pd.isna(data_sample1.iloc[i, 0]) == True:
            cnt1 += 1

    d1 = sorted(d1)  ########### 클수록 좋으면 Reverse = True 추가

    r1 = d1.index(float(graph_score)) + cnt1 + 1

    f_score = 1 - generalization(graph_score, d1)

    print(f'Grpah 순위 = {r1}/{len(d1)}')