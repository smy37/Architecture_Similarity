#### text_sub에 있는 BoW를 이용하여 946개의 비교값을 생성함.
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-mpnet-base-v2')
from scipy import spatial
import re

p = re.compile('[^a-zA-Z0-9_ ]')

# from Sim_App import text_sub as tx
import csv
import os


path_1 = r'C:\Users\seong\Desktop\ArchSim'
A = os.listdir(path_1)
with open(f'trans_tx_result.csv', 'w', encoding='cp949', newline='') as wd:
    wr = csv.writer(wd)
    wr.writerow(
        ['', 'Concept(BoW)'])
    for x in range(1,12):               ### 1) 11개의 사조들을 변수 x로 설정!
        b_group3 = os.listdir(r'C:\Users\seong\Desktop\ArchSim\{}'.format(A[x-1]))
        for k in range(4):              ### 2) 각 사조별로 4개씩의 건물이 존재하고 이를 변수 k로 설정!
            depth_1 = 'C:/Users/seong/Desktop/ArchSim/{}/{}'.format(A[x - 1], b_group3[k])
            temp = os.listdir(depth_1+'/Text')
            f1 = open(depth_1+'/Text'+'/'+temp[0], 'r', encoding='UTF8')
            sen1 = f1.readline().lower().replace(',', '').replace('.', '')
            sen1 = p.sub('', sen1)
            for t in range(k+1,4):      ### 3) 같은 사조의 건물들끼리 비교하기위해 k와 pair를 이루는 변수 t 설정!
                depth_2 = 'C:/Users/seong/Desktop/ArchSim/{}/{}'.format(A[x - 1], b_group3[t])
                temp2 = os.listdir(depth_2+'/Text')

                print(depth_1)
                print(depth_2)

                f2 = open(depth_2+'/Text' + '/' + temp2[0], 'r', encoding='UTF8')

                sen2 = f2.readline().lower().replace(',', '').replace('.', '')

                sen2 = p.sub('', sen2)
                vectors = model.encode([sen1, sen2])
                similarity = 1 - spatial.distance.cosine(vectors[0], vectors[1])
                wr.writerow([b_group3[k] + '&' + b_group3[t], similarity])

                f2.close()
            f1.close()
        for y in range(x+1, 12):        ### 4) 위에서 설정한 x와 다른 사조들의 쌍을 변수 y로 설정!
            era_1 = x
            era_2 = y
            b_group1 = os.listdir(r'C:\Users\seong\Desktop\ArchSim\{}'.format(A[era_1-1]))
            b_group2 = os.listdir(r'C:\Users\seong\Desktop\ArchSim\{}'.format(A[era_2-1]))
            for i in range(4):          ### 4) 2)에서의 과정과 동일!

                depth_1 = 'C:/Users/seong/Desktop/ArchSim/{}/{}'.format(A[era_1 - 1], b_group1[i])
                temp = os.listdir(depth_1+'/Text')
                f1 = open(depth_1+'/Text' + '/' + temp[0], 'r', encoding='UTF8')

                sen1 = f1.readline().lower().replace(',', '').replace('.', '')
                sen1 = p.sub('', sen1)
                for j in range(4):       ### 4) 3)에서의 과정과 동일!
                    depth_2 = 'C:/Users/seong/Desktop/ArchSim/{}/{}'.format(A[era_2 - 1], b_group2[j])
                    temp2 = os.listdir(depth_2+'/Text')
                    print(depth_1)
                    print(depth_2)
                    f2 = open(depth_2+'/Text' + '/' + temp2[0], 'r', encoding='UTF8')

                    sen2 = f2.readline().lower().replace(',', '').replace('.', '')
                    sen2 = p.sub('', sen2)
                    vectors = model.encode([sen1, sen2])
                    similarity = 1 - spatial.distance.cosine(vectors[0], vectors[1])
                    wr.writerow([b_group3[k] + '&' + b_group3[t], similarity])
                    f2.close()
