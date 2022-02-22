from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import os
from nltk.tokenize import word_tokenize
import re
from scipy import spatial



# print('bag of words vector :',vect.fit_transform(text).toarray())
# print('vocabulary :',vect.vocabulary_)

def bow(sentence1, sentence2):
    p = re.compile('[^a-zA-Z0-9_ ]')

    sentence1 = p.sub('', sentence1)
    sentence2 = p.sub('', sentence2)

    stemmer = PorterStemmer()
    stop_words = stopwords.words("english")
    vect = CountVectorizer(stop_words=stop_words)
    bag1 = [stemmer.stem(w) for w in word_tokenize(sentence1)]
    bag2 = [stemmer.stem(w) for w in word_tokenize(sentence2)]

    bags = sorted(bag1+bag2)

    vec1 = {}
    vec2 = {}

    for i in bags:
        vec1[i] = 0
        vec2[i] = 0

    for i in bag1:
        vec1[i] += 1

    for i in bag2:
        vec2[i] += 1

    dist_1 = 1 - spatial.distance.cosine(list(vec1.values()), list(vec2.values()))
    
    return dist_1


if __name__ == '__main__':
    in_path = 'C:/Users/seong/Desktop/app_front/text/'

    file_list = os.listdir(in_path)

    f1 = open(in_path + f'/{file_list[0]}', 'r', encoding='UTF8')
    f2 = open(in_path + f'/{file_list[1]}', 'r', encoding='UTF8')

    sen1 = f1.readline().lower().replace(',', '').replace('.', '')
    sen2 = f2.readline().lower().replace(',', '').replace('.', '')

    stemmer = PorterStemmer()
    stop_words = stopwords.words("english")
    vect = CountVectorizer(stop_words=stop_words)
    print(bow(sen1,sen2))