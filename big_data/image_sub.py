### 본 코드는 VGG16 기반의 Feature Extraction을 통한 대용량 이미지 유사도 비교 코드임.


import os
import csv
import image_sub2 as ds
import sys

path_1 = r'C:\Users\seong\Desktop\ArchSim'
A = os.listdir(path_1)

data_category = ['External', 'Internal', 'Plan', 'Text']


with open(f'new_result.csv', 'w', encoding='cp949', newline='') as wd:
    wr = csv.writer(wd)
    wr.writerow(
        ['', 'External(VGG16)', 'Internal(VGG16)', 'Plan(VGG16)'])

    for x in range(1, 12):
        b_group3 = os.listdir(r'C:\Users\seong\Desktop\ArchSim\{}'.format(A[x - 1]))
        for k in range(4):
            depth_1 = 'C:/Users/seong/Desktop/ArchSim/{}/{}'.format(A[x - 1], b_group3[k])

            for t in range(k + 1, 4):
                depth_2 = 'C:/Users/seong/Desktop/ArchSim/{}/{}'.format(A[x - 1], b_group3[t])
                print(depth_1)
                print(depth_2)
                result_d = ds.d_search(depth_1, depth_2)
                print(result_d)
                print(result_d['E_Image']['DEEP'])
                wr.writerow([b_group3[k] + '&' + b_group3[t], result_d['E_Image']['DEEP'], result_d['I_Image']['DEEP'], result_d['P_Image']['DEEP']])


        for y in range(x + 1, 12):
            era_1 = x
            era_2 = y
            b_group1 = os.listdir(r'C:\Users\seong\Desktop\ArchSim\{}'.format(A[era_1 - 1]))
            b_group2 = os.listdir(r'C:\Users\seong\Desktop\ArchSim\{}'.format(A[era_2 - 1]))
            for i in range(4):
                depth_1 = 'C:/Users/seong/Desktop/ArchSim/{}/{}'.format(A[era_1 - 1], b_group1[i])

                for j in range(4):
                    depth_2 = 'C:/Users/seong/Desktop/ArchSim/{}/{}'.format(A[era_2 - 1], b_group2[j])

                    print(depth_1)
                    print(depth_2)

                    result_d = ds.d_search(depth_1, depth_2)
                    wr.writerow([b_group3[k] + '&' + b_group3[t], result_d['E_Image']['DEEP'], result_d['I_Image']['DEEP'], result_d['P_Image']['DEEP']])