import json
import os
import networkx as nx
import subprocess
import csv
import numpy as np
from numpy import dot
from numpy.linalg import norm
from scipy import spatial
import sys


def graph_embedding(A,cnt):
    json_dic1 = {'edges': list(A.edges())}

    with open(f'C:/Users/seong/PycharmProjects/First_Env/Sim_App/graph2vec/dataset/{cnt}.json', 'w') as wr:
        json.dump(json_dic1, wr)

def cos_sim(A, B):
    return dot(A, B) / (norm(A) * norm(B))


def l2_distance(A, B):
    return (((A - B) ** 2).sum()) ** 0.5


if __name__ == '__main__':

    temp_path = 'C:/Users/seong/Desktop/apartment/'
    in_path = os.listdir(temp_path)
    cnt = 0
    for i in in_path:
        with open(temp_path+ i, 'r') as r_json_file:
            temp_data = json.load(r_json_file)
        g1 = nx.node_link_graph(temp_data)
        graph_embedding(g1,cnt)
        cnt+=1


    subprocess.call(['python', 'src/graph2vec.py', '--dimensions', '128', '--epochs', '10'], shell=True, cwd='graph2vec/')

    result_vec = []

    with open(r'C:\Users\seong\PycharmProjects\First_Env\Sim_App\graph2vec\features\nci1.csv', 'r') as rd:
        data = csv.reader(rd)
        cnt = 0
        for i in data:
            if cnt >= 1:
                result_vec.append(list(map(float, i[1:])))
            cnt += 1

    print(result_vec)
    temp = []
    for i in range(len(in_path)):
        for j in range(i+1, len(in_path)):
            cos_v = 1 - spatial.distance.cosine(result_vec[i], result_vec[j])
            temp.append([in_path[i],in_path[j],cos_v])
            print(in_path[i],in_path[j],cos_v)
    print(sorted(temp, key = lambda x: x[2]))
    sys.exit()
    first = np.array(result_vec[0])
    second = np.array(result_vec[1])
    cos_v = 1 - spatial.distance.cosine(result_vec[0], result_vec[1])

    print(cos_v)
