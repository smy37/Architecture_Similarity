import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("seaborn")

data = pd.read_csv(r'C:\Users\seong\Desktop\ArchitectureSimilarity\2022_result_general.csv')

data_sample = data.iloc[:, 3:]
model = KMeans(n_clusters=2)
model.fit(data_sample)
temp_list = model.labels_

method_2 = {}
method_3 = []
for i in range(len(data)):
    tt = data.iloc[i, 1]
    if tt == data.iloc[i,2]:
        if tt[3:] not in method_2:
            method_2[tt[3:]] = [i]
        else:
            method_2[tt[3:]].append(i)
    else:
        method_3.append(i)
print(method_2)


for i in method_2:
    cnt1 = 0
    cnt2 = 0
    cnt3 = 0
    for j in method_2[i]:
        if temp_list[j] == 0:
            cnt1 += 1
        elif temp_list[j] == 1:
            cnt2 += 1

    print(f'Era = {i} : cluster1 = {cnt1}, cluster2 = {cnt2}')
