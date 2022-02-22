import cv2
import os
import numpy as np
import PIL.Image as Image
import pandas as pd


### 1. MSE & SSIM Define
from skimage.metrics import structural_similarity as compare_ssim

def mse(A,B):
    err = np.sum((A.astype(float)-B.astype(float))**2)
    err = err/(A.shape[0]*A.shape[1])
    return err

def img_preprocess(A: str, B: str):

    Uimg_1 = Image.open(A)
    Cimg_1 = Image.open(B)
    back = Image.open(r'C:\Users\seong\PycharmProjects\First_Env\Sim_App\background\white.png')

    back2 = back.resize(Uimg_1.size)
    cri = min(Uimg_1.width / Cimg_1.width, Uimg_1.height / Cimg_1.height)
    sam_2 = Cimg_1.resize((round(cri * Cimg_1.width), round(cri * Cimg_1.height)))
    a = back2.width - sam_2.width
    b = back2.height - sam_2.height
    if a == 0:
        cri2 = ('height', sam_2.height / 2)
    elif b == 0:
        cri2 = ('width', sam_2.width / 2)
    if cri2[0] == 'height':
        start = int(back2.height / 2 - cri2[1])
        end = start + sam_2.height
        back2.paste(sam_2, (0, start, sam_2.width, end))
    elif cri2[0] == 'width':
        start = int(back2.width / 2 - cri2[1])
        end = start + sam_2.width
        back2.paste(sam_2, (start, 0, end, sam_2.height))
    back2.save(r'C:\Users\seong\Desktop\app_front\img_preprocess\new1.png')
    return back2


### 2. Feature Extraction Define
from scipy import spatial
from tqdm import tqdm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

data_category = ['External', 'Internal', 'Plan']


class FeatureExtractor:
    def __init__(self):
        # Use VGG-16 as the architecture and ImageNet for the weight
        base_model = VGG16(weights='imagenet')
        # Customize the model to return features from fully-connected layer
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
    def extract(self, img):
        # Resize the image
        img = img.resize((224, 224))
        # Convert the image color space
        img = img.convert('RGB')
        # Reformat the image
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        # Extract Features
        feature = self.model.predict(x)[0]
        return feature / np.linalg.norm(feature)
    def get_feature(self,image_data:list):
        self.image_data = image_data
        #fe = FeatureExtractor()
        features = []
        for img_path in tqdm(self.image_data): # Iterate through images
            # Extract Features
            try:
                feature = self.extract(img=Image.open(img_path))
                features.append(feature)
            except:
                features.append(None)
                continue
        return features

def get_query_vector(image_path: str, fe):
    img = Image.open(image_path)
    query_vector = fe.extract(img)
    return query_vector

f_extract = FeatureExtractor()


### 3. Utils Define
def generalization(target, database):
    tt = (target - np.min(database)) / (np.max(database)-np.min(database))

    return tt

def image_calculation(fname1, fname2, sta, f_extraction):
    f_extract = f_extraction

    img1 = cv2.imread(fname1)

    img_preprocess(fname1, fname2)  ##### PIL과 opencv의 차이 때문에 저장과 로드가 들어간다.

    img2 = cv2.imread(r'C:\Users\seong\Desktop\app_front\img_preprocess\new1.png')

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    #### SSIM과 MSE 수행
    (score, diff) = compare_ssim(gray1, gray2, full=True)
    score2 = mse(gray1, gray2)



    #### Deep Learning 기반의 Feature Extraction 수행
    a = get_query_vector(fname1, f_extract)
    b = get_query_vector(fname2, f_extract)
    score3 = 1 - spatial.distance.cosine(a, b)

    #### 데이터 베이스의 값들과 비교하여 Rank 산정하기
    data = pd.read_csv('./result/result_final.csv')

    if sta == 'exterior':
        data_sample1 = data['External(SSIM)']
        data_sample2 = data['External(MSE)']
        data_sample3 = data['External(VGG16)']
    elif sta == 'interior':
        data_sample1 = data['Internal(SSIM)']
        data_sample2 = data['Internal(MSE)']
        data_sample3 = data['Internal(VGG16)']
    elif sta == 'plan':
        data_sample1 = data['Plan(SSIM)']
        data_sample2 = data['Plan(MSE)']
        data_sample3 = data['Plan(VGG16)']

    s1 = pd.DataFrame([score * 100])
    s2 = pd.DataFrame([score2 / 1000])
    s3 = pd.DataFrame([score3])

    data_sample1 = pd.concat([data_sample1, s1])
    data_sample2 = pd.concat([data_sample2, s2])
    data_sample3 = pd.concat([data_sample3, s3])

    cnt1 = 0
    cnt2 = 0
    cnt3 = 0

    d1 = []
    d2 = []
    d3 = []
    for i in range(len(data_sample1)):
        if pd.isna(data_sample1.iloc[i, 0]) == False:
            d1.append(data_sample1.iloc[i, 0])
        elif pd.isna(data_sample1.iloc[i, 0]) == True:
            cnt1 += 1

        if pd.isna(data_sample2.iloc[i, 0]) == False:
            d2.append(data_sample2.iloc[i, 0])
        elif pd.isna(data_sample2.iloc[i, 0]) == True:
            cnt2 += 1

        if pd.isna(data_sample3.iloc[i, 0]) == False:
            d3.append(data_sample3.iloc[i, 0])
        elif pd.isna(data_sample3.iloc[i, 0]) == True:
            cnt3 += 1

    print(f'Null 값들, {cnt1},{cnt2},{cnt3}')
    d1 = sorted(d1, reverse=True)
    d2 = sorted(d2)
    d3 = sorted(d3, reverse=True)


    r1 = d1.index(float(score * 100)) + 1
    r2 = d2.index(float(score2 / 1000)) + 1
    r3 = d3.index(float(score3)) + 1

    print(f'SSIM 값 = {score:.5f}, 순위 = {r1}/{len(d1) + cnt1}')
    print(f'MSE 값 = {score2:.5f}, 순위 = {r2}/{len(d2) + cnt2}')
    print(f'VGG16 값 = {score3:.5f}, 순위 = {r3}/{len(d3) + cnt3}')

    es1 = generalization(score * 100, d1)
    es2 = 1 - generalization(score2 / 1000, d2)
    es3 = generalization(score3, d3)

    temp_a = f'SSIM 값 = {score:.5f}, 순위 = {r1}/{len(d1) + cnt1}'
    temp_b = f'MSE 값 = {score2:.5f}, 순위 = {r2}/{len(d2) + cnt2}'
    temp_c = f'VGG16 값 = {score3:.5f}, 순위 = {r3}/{len(d3) + cnt3}'

    return es1, es2, es3, f'{r1}/{len(d1) + cnt1}', f'{r2}/{len(d2) + cnt2}', f' {r3}/{len(d3) + cnt3}'



if __name__ == '__main__':
    temp_path = 'C:/Users/seong/Desktop/app_front/image/'

    set = ['External', 'Internal', 'Plan']
    f_extract = FeatureExtractor()


    for x in set:
        in_path = temp_path + f'/{x}/'
        file_list = os.listdir(in_path)

        if x == 'External':
            #### 이미지 불러오기와 전처리 단계
            img1 = cv2.imread(in_path + file_list[0])
            img_preprocess(in_path + file_list[0], in_path + file_list[1])  ##### PIL과 opencv의 차이 때문에 저장과 로드가 들어간다.
            img2 = cv2.imread(r'C:\Users\seong\Desktop\app_front\img_preprocess\new1.png')

            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


            #### SSIM과 MSE 수행
            (score, diff) = compare_ssim(gray1, gray2, full=True)
            score2 = mse(gray1, gray2)

            #### Deep Learning 기반의 Feature Extraction 수행
            a = get_query_vector(in_path + file_list[0], f_extract)
            b = get_query_vector(in_path + file_list[1], f_extract)
            score3 = 1 - spatial.distance.cosine(a, b)

            #### 데이터 베이스의 값들과 비교하여 Rank 산정하기
            data = pd.read_csv('result/result_final.csv')

            data_sample1 = data['External(SSIM)']
            data_sample2 = data['External(MSE)']
            data_sample3 = data['External(VGG16)']

            s1 = pd.DataFrame([score * 100])
            s2 = pd.DataFrame([score2 / 1000])
            s3 = pd.DataFrame([score3])

            data_sample1 = pd.concat([data_sample1, s1])
            data_sample2 = pd.concat([data_sample2, s2])
            data_sample3 = pd.concat([data_sample3, s3])

            cnt1 = 0
            cnt2 = 0
            cnt3 = 0

            d1 = []
            d2 = []
            d3 = []
            for i in range(len(data_sample1)):
                if pd.isna(data_sample1.iloc[i, 0]) == False:
                    d1.append(data_sample1.iloc[i, 0])
                elif pd.isna(data_sample1.iloc[i, 0]) == True:
                    cnt1 += 1

                if pd.isna(data_sample2.iloc[i, 0]) == False:
                    d2.append(data_sample2.iloc[i, 0])
                elif pd.isna(data_sample2.iloc[i, 0]) == True:
                    cnt2 += 1

                if pd.isna(data_sample3.iloc[i, 0]) == False:
                    d3.append(data_sample3.iloc[i, 0])
                elif pd.isna(data_sample3.iloc[i, 0]) == True:
                    cnt3 += 1

            print(f'Null 값들, {cnt1},{cnt2},{cnt3}')
            d1 = sorted(d1, reverse = True)
            d2 = sorted(d2)
            d3 = sorted(d3, reverse = True)
            print(d1)
            print(d2)
            print(d3)

            r1 = d1.index(float(score * 100))  + 1
            r2 = d2.index(float(score2 / 1000)) + 1
            r3 = d3.index(float(score3)) + 1

            print(f'{x} SSIM 값 = {score:.5f}, 순위 = {r1}/{len(d1)+cnt1}')
            print(f'{x} MSE 값 = {score2:.5f}, 순위 = {r2}/{len(d2)+cnt2}')
            print(f'{x} VGG16 값 = {score3:.5f}, 순위 = {r3}/{len(d3)+cnt3}')

            es1 = generalization(score*100, d1)
            es2 = 1 - generalization(score2/1000, d2)
            es3 = generalization(score3, d3)


        elif x == 'Internal':
            #### 이미지 불러오기와 전처리 단계
            img1 = cv2.imread(in_path + file_list[0])
            img_preprocess(in_path + file_list[0], in_path + file_list[1])  ##### PIL과 opencv의 차이 때문에 저장과 로드가 들어간다.
            img2 = cv2.imread(r'C:\Users\seong\Desktop\app_front\img_preprocess\new1.png')

            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


            #### SSIM과 MSE 수행
            (score, diff) = compare_ssim(gray1, gray2, full=True)
            score2 = mse(gray1, gray2)

            #### Deep Learning 기반의 Feature Extraction 수행
            a = get_query_vector(in_path + file_list[0], f_extract)
            b = get_query_vector(in_path + file_list[1], f_extract)
            score3 = 1 - spatial.distance.cosine(a, b)

            #### 데이터 베이스의 값들과 비교하여 Rank 산정하기
            data = pd.read_csv('./result/result_final.csv')

            data_sample1 = data['Internal(SSIM)']
            data_sample2 = data['Internal(MSE)']
            data_sample3 = data['Internal(VGG16)']

            s1 = pd.DataFrame([score * 100])
            s2 = pd.DataFrame([score2 / 1000])
            s3 = pd.DataFrame([score3])

            data_sample1 = pd.concat([data_sample1, s1])
            data_sample2 = pd.concat([data_sample2, s2])
            data_sample3 = pd.concat([data_sample3, s3])

            cnt1 = 0
            cnt2 = 0
            cnt3 = 0

            d1 = []
            d2 = []
            d3 = []
            for i in range(len(data_sample1)):
                if pd.isna(data_sample1.iloc[i, 0]) == False:
                    d1.append(data_sample1.iloc[i, 0])
                elif pd.isna(data_sample1.iloc[i, 0]) == True:
                    cnt1 += 1
                if pd.isna(data_sample2.iloc[i, 0]) == False:
                    d2.append(data_sample2.iloc[i, 0])
                elif pd.isna(data_sample2.iloc[i, 0]) == True:
                    cnt2 += 1

                if pd.isna(data_sample3.iloc[i, 0]) == False:
                    d3.append(data_sample3.iloc[i, 0])
                elif pd.isna(data_sample3.iloc[i, 0]) == True:
                    cnt3 += 1
            print(f'Null 값들, {cnt1},{cnt2},{cnt3}')

            d1 = sorted(d1, reverse= True)
            d2 = sorted(d2)
            d3 = sorted(d3, reverse= True)

            print(d1)
            print(d2)
            print(d3)


            r1 = d1.index(float(score * 100))  + 1
            r2 = d2.index(float(score2 / 1000)) + 1
            r3 = d3.index(float(score3))  + 1

            print(f'{x} SSIM 값 = {score:.5f}, 순위 = {r1}/{len(d1)+cnt1}')
            print(f'{x} MSE 값 = {score2:.5f}, 순위 = {r2}/{len(d2)+cnt2}')
            print(f'{x} VGG16 값 = {score3:.5f}, 순위 = {r3}/{len(d3)+cnt3}')

            is1 = generalization(score*100, d1)
            is2 = 1 - generalization(score2/1000, d2)
            is3 = generalization(score3, d3)

        elif x == 'Plan':
            #### 이미지 불러오기와 전처리 단계
            img1 = cv2.imread(in_path + file_list[0])
            img_preprocess(in_path + file_list[0], in_path + file_list[1])  ##### PIL과 opencv의 차이 때문에 저장과 로드가 들어간다.
            img2 = cv2.imread(r'C:\Users\seong\Desktop\app_front\img_preprocess\new1.png')

            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # ret, dst = cv2.threshold(gray2, 200, 255, cv2.THRESH_BINARY)
            # cv2.imwrite(in_path + "/testp2.png", dst)
            # ret2, dst2 = cv2.threshold(gray1, 130, 255, cv2.THRESH_BINARY)
            # cv2.imwrite(in_path + "/testp.png", dst2)
            # cv2.imwrite(in_path + '/test1.png', gray1)
            # cv2.imwrite(in_path + '/test2.png', gray2)

            #### SSIM과 MSE 수행
            (score, diff) = compare_ssim(gray1, gray2, full=True)
            score2 = mse(gray1, gray2)

            #### Deep Learning 기반의 Feature Extraction 수행
            a = get_query_vector(in_path + file_list[0], f_extract)
            b = get_query_vector(in_path + file_list[1], f_extract)
            score3 = 1 - spatial.distance.cosine(a, b)


            #### 데이터 베이스의 값들과 비교하여 Rank 산정하기
            data = pd.read_csv('./result/result_final.csv')

            data_sample1 = data['Plan(SSIM)']
            data_sample2 = data['Plan(MSE)']
            data_sample3 = data['Plan(VGG16)']

            s1 = pd.DataFrame([score * 100])
            s2 = pd.DataFrame([score2 / 1000])
            s3 = pd.DataFrame([score3])

            data_sample1 = pd.concat([data_sample1, s1])
            data_sample2 = pd.concat([data_sample2, s2])
            data_sample3 = pd.concat([data_sample3, s3])

            cnt1 = 0
            cnt2 = 0
            cnt3 = 0

            d1 = []
            d2 = []
            d3 = []
            for i in range(len(data_sample1)):
                if pd.isna(data_sample1.iloc[i, 0]) == False:
                    d1.append(data_sample1.iloc[i, 0])
                elif pd.isna(data_sample1.iloc[i, 0]) == True:
                    cnt1 += 1

                if pd.isna(data_sample2.iloc[i, 0]) == False:
                    d2.append(data_sample2.iloc[i, 0])
                elif pd.isna(data_sample2.iloc[i, 0]) == True:
                    cnt2 += 1

                if pd.isna(data_sample3.iloc[i, 0]) == False:
                    d3.append(data_sample3.iloc[i, 0])
                elif pd.isna(data_sample3.iloc[i, 0]) == True:
                    cnt3 += 1
            print(f'Null 값들, {cnt1},{cnt2},{cnt3}')

            d1 = sorted(d1, reverse=True)
            d2 = sorted(d2)
            d3 = sorted(d3, reverse=True)

            print(d1)
            print(d2)
            print(d3)
            r1 = d1.index(float(score * 100)) + 1
            r2 = d2.index(float(score2 / 1000)) + 1
            r3 = d3.index(float(score3)) + 1

            print(f'{x} SSIM 값 = {score:.5f}, 순위 = {r1}/{len(d1)+cnt1}')
            print(f'{x} MSE 값 = {score2:.5f}, 순위 = {r2}/{len(d2)+cnt2}')
            print(f'{x} VGG16 값 = {score3:.5f}, 순위 = {r3}/{len(d3)+cnt3}')

            ps1 = generalization(score*100, d1)
            ps2 = 1 - generalization(score2/1000, d2)
            ps3 = generalization(score3, d3)



