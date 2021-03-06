# __*Architecture_Similarity*__
## 사람들이 두개의 건물을 놓고 비교할 때 유사성 판단의 기준이 되는 것은 다음 4가지와 같다. 내관, 외관, 공간 구성, 컨셉.

### 건축 분야에서 위의 4가지를 포함하고 있는 데이터는 다음과 같다. 
- 1. 내관: 사진, 렌더링 이미지, 단면, 투시도, 3d 모델 
- 2. 외관: 사진, 렌더링 이미지, 입면, 조감도, 3d 모델 
- 3. 공간구성: 평면, 단면, 캐드, 3d 모델
- 4. 컨셉: 기사, 설명글

### 위에서 언급된 아날로그 데이터들을 디지털화, 수치화 했을때의 결과물은 다음과 같다. (캐드나 3d 모델의 경우에는 이미 수치화된 디지털 데이터이다.)  
- 1. 사진, 렌더링 이미지, 평면, 단면, 투시도: RGB 행렬 
- 2. 평면, 단면: 그래프 
- 3. 기사, 설명글: 디지털 텍스트(글자도 결국 컴퓨터에서는 미리 정해진 상수들의 결합일 뿐이다.)

### 윗 단계에서 적용 가능한 알고리즘이 존재하고 임베딩을 시켜 비슷한 것은 비슷한 것끼리 모아주는 것도 가능하다. 
- 1. RGB 행렬
  - 1.1 픽셀 레벨 단계 알고리즘: MSE, SSIM
  - 1.2 임베딩: Feature Extraction(ex. Using VGG16)
- 2. 그래프
  - 2.1 그래프 단계 알고리즘: GED
  - 2.2 임베딩: graph2vec
- 3. 디지털 텍스트 
  - 3.1 통계에 근거한 변환: Bag of Words
  - 3.2 임베딩: sent2vec

