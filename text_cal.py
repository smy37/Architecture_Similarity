from scipy import spatial
import os.path
import numpy as np
import pandas as pd
import re
import text_sub
from sentence_transformers import SentenceTransformer

def generalization(target, database):
    tt = (target - np.min(database)) / (np.max(database)-np.min(database))

    return tt

def text_calculation(fname1, fname2):
    f1 = open(fname1, 'r', encoding='UTF8')
    f2 = open(fname2, 'r', encoding='UTF8')

    p = re.compile('[^a-zA-Z0-9_ ]')

    sen1 = ''.join(f1.readlines())
    sen2 = ''.join(f2.readlines())

    bow_score = text_sub.bow(sen1, sen2)

    sen1 = p.sub('', sen1)
    sen2 = p.sub('', sen2)

    sentences = []

    sentences.append(sen1)
    sentences.append(sen2)

    model = SentenceTransformer('all-mpnet-base-v2')
    sen1 = p.sub('', sen1)
    sen2 = p.sub('', sen2)
    vectors = model.encode([sen1, sen2])
    dist_1 = 1 - spatial.distance.cosine(vectors[0], vectors[1])
    # vectorizer = Vectorizer()
    # vectorizer.bert(sentences)
    # vectors_vert = vectorizer.vectors
    # dist_1 = 1 - spatial.distance.cosine(vectors_vert[0], vectors_vert[1])

    print(f'Text Similarity = {dist_1}')
    print(f'Bow Similarity = {bow_score}')

    data = pd.read_csv('./result/result_final.csv')
    data_sample1 = data['Concept(Transformer)']
    data_sample2 = data['Concept(BoW)']

    s1 = pd.DataFrame([dist_1])
    s2 = pd.DataFrame([bow_score])

    data_sample1 = pd.concat([data_sample1, s1])
    data_sample2 = pd.concat([data_sample2, s2])
    cnt1 = 0
    cnt2 = 0
    d1 = []
    d2 = []
    for i in range(len(data_sample1)):
        if pd.isna(data_sample1.iloc[i, 0]) == False:
            d1.append(data_sample1.iloc[i, 0])
        elif pd.isna(data_sample1.iloc[i, 0]) == True:
            cnt1 += 1
        if pd.isna(data_sample2.iloc[i, 0]) == False:
            d2.append(data_sample2.iloc[i, 0])
        elif pd.isna(data_sample2.iloc[i, 0]) == False:
            cnt2 += 1

    d1 = sorted(d1, reverse=True)
    d2 = sorted(d2, reverse=True)

    r1 = d1.index(float(dist_1)) + 1
    r2 = d2.index(float(bow_score)) + 1

    f_score = generalization(dist_1, d1)
    f_score2 = generalization(bow_score, d2)
    print(f'Text 순위 = {r1}/{len(d1) + cnt1}')
    print(f'BoW 순위 = {r2}/{len(d2) + cnt2}')

    return f_score, f_score2, f'{r1}/{len(d1) + cnt1}', f'{r2}/{len(d2) + cnt2}'


def text_plot(A,B):
    with open(A, 'r', encoding='UTF8') as tt:
        temp_t = tt.readlines()
        real_temp = ''
        cnt = 0
        cnt2 = 0
        for i in temp_t:
            for j in i:
                cnt += 1
                real_temp += j
                if cnt > 20:
                    real_temp += '\n'
                    cnt = 0
                    cnt2 += 1
                if cnt2 > 10:
                    break
        f_a = real_temp
    with open(B, 'r', encoding='UTF8') as tt:
        temp_t = tt.readlines()
        real_temp = ''
        cnt = 0
        cnt2 = 0
        for i in temp_t:
            for j in i:
                cnt += 1
                real_temp += j
                if cnt > 20:
                    real_temp += '\n'
                    cnt = 0
                    cnt2 += 1
                if cnt2 > 10:
                    break
        f_b = real_temp
    return f_a, f_b

if __name__ == '__main__':
    in_path = 'C:/Users/seong/Desktop/app_front/text/'

    file_list = os.listdir(in_path)
    f1 = open(in_path + f'/{file_list[0]}', 'r', encoding='UTF8')
    f2 = open(in_path + f'/{file_list[1]}', 'r', encoding='UTF8')
    p = re.compile('[^a-zA-Z0-9_ ]')


    sen1 = ''.join(f1.readlines())
    sen2 = ''.join(f2.readlines())

    bow_score = text_sub.bow(sen1, sen2)

    sen1 = p.sub('', sen1)
    sen2 = p.sub('', sen2)

    sentences = []

    sentences.append(sen1)
    sentences.append(sen2)

    # vectorizer = Vectorizer()
    # vectorizer.bert(sentences)
    # vectors_vert = vectorizer.vectors
    model = SentenceTransformer('all-mpnet-base-v2')
    sen1 = p.sub('', sen1)
    sen2 = p.sub('', sen2)
    vectors = model.encode([sen1, sen2])
    dist_1 = 1 - spatial.distance.cosine(vectors[0], vectors[1])
    # dist_1 = 1 - spatial.distance.cosine(vectors_vert[0], vectors_vert[1])
    print(f'Text Similarity = {dist_1:.3f}')
    print(f'Bow Similarity = {bow_score:.3f}')


    data = pd.read_csv('./result/result_final.csv')
    data_sample1 = data['Concept(Transformer)']
    data_sample2 = data['Concept(BoW)']

    s1 = pd.DataFrame([dist_1])
    s2 = pd.DataFrame([bow_score])

    data_sample1 = pd.concat([data_sample1, s1])
    data_sample2 = pd.concat([data_sample2, s2])
    cnt1 = 0
    cnt2 = 0
    d1 = []
    d2 = []
    for i in range(len(data_sample1)):
        if pd.isna(data_sample1.iloc[i, 0]) == False:
            d1.append(data_sample1.iloc[i, 0])
        elif pd.isna(data_sample1.iloc[i, 0]) == True:
            cnt1 += 1
        if pd.isna(data_sample2.iloc[i,0]) == False:
            d2.append(data_sample2.iloc[i,0])
        elif pd.isna(data_sample2.iloc[i,0]) == False:
            cnt2 += 1

    d1 = sorted(d1, reverse=True)
    d2 = sorted(d2, reverse = True)

    r1 = d1.index(float(dist_1)) + 1
    r2 = d2.index(float(bow_score)) + 1

    f_score = generalization(dist_1, d1)
    f_score2 = generalization(bow_score, d2)
    print(f'Text 순위 = {r1}/{len(d1)+cnt1}')
    print(f'BoW 순위 = {r2}/{len(d2)+cnt2}')