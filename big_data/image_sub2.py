### 본 코드는 VGG16 기반의 Feature Extraction을 통한 대용량 이미지 유사도 비교 코드임.


import DeepImageSearch as dis
from scipy import spatial
import os
from PIL import Image
import numpy as np
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

def d_search(tar_1, tar_2):
    result = {}
    e_tar_1 = tar_1 + '/External'
    e_tar_2 = tar_2 + '/External'
    i_tar_1 = tar_1 + '/Internal'
    i_tar_2 = tar_2 + '/Internal'
    p_tar_1 = tar_1 + '/Plan'
    p_tar_2 = tar_2 + '/Plan'
    #################

    E_files_1 = os.listdir(e_tar_1)
    E_files_2 = os.listdir(e_tar_2)
    I_files_1 = os.listdir(i_tar_1)
    I_files_2 = os.listdir(i_tar_2)
    P_files_1 = os.listdir(p_tar_1)
    P_files_2 = os.listdir(p_tar_2)

    #### 1. External Image Similarity
    temp1 = 0

    for i in E_files_1:
        a = get_query_vector(e_tar_1 + f'/{i}', f_extract)
        for j in E_files_2:

            b = get_query_vector(e_tar_2 + f'/{j}', f_extract)
            temp1 += (1 - spatial.distance.cosine(a, b))

    result['E_Image'] = {'DEEP':temp1/(len(E_files_1)*len(E_files_2))}

    #### 2. Internal Image Similarity
    temp1 = 0

    for i in I_files_1:
        a = get_query_vector(i_tar_1 + f'/{i}', f_extract)
        for j in I_files_2:

            b = get_query_vector(i_tar_2 + f'/{j}', f_extract)
            temp1 += (1-spatial.distance.cosine(a,b))
    result['I_Image'] = {'DEEP':temp1/(len(I_files_1)*len(I_files_2))}

    #### 3. Plan Image Similarity
    temp1 = 0

    for i in P_files_1:
        a = get_query_vector(p_tar_1 + f'/{i}', f_extract)
        for j in P_files_2:
            b = get_query_vector(p_tar_2 + f'/{j}', f_extract)
            temp1 += (1 - spatial.distance.cosine(a, b))
    result['P_Image'] = {'DEEP':temp1/(len(P_files_1)*len(P_files_2))}


    return result