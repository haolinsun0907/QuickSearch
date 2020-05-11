from Modules import load_stopwords,stemming_f,stopword_removal_f,normalization_f
from nltk import PorterStemmer
import os
import re
import nltk
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from random import randint,sample
from sklearn.externals import joblib

corpus_path="./reuters21578"
stopwords=load_stopwords()
stemmer=PorterStemmer()

def get_corpus():
    corpus=[]
    for filename in os.listdir(corpus_path):
        if filename.startswith("reut2"):
            corpus+=parse_content(corpus_path+"/"+filename)
    # for c in corpus:
    #     print(c)
    return corpus

def parse_content(file_path):
    content=[]
    with open(file_path,encoding="utf-8") as f:
        s=f.read()
        article=s.split("</REUTERS>")
        for a in article[:-1]:
            if "<TEXT>" not in a:
                continue
            text=a.split("<TEXT>")[1]
            text=re.sub("<.*?>|&.*?;","",text).strip().lower()
            topic=re.search(r"<TOPICS>(.*?)</TOPICS>",a).groups(1)[0]
            if topic:
                topic=topic.replace("<D>","")
                topic = topic.replace("</D>", " ")
            topic=topic.split()
            content.append({"text":text,"topic":topic})
    return content

def process(text):
    text=text.replace("\n"," ")
    words=nltk.word_tokenize(text)
    words = normalization_f(words)
    words = stopword_removal_f(words, stopwords)
    words = stemming_f(words, stemmer)
    return words


def build_dict(corpus):
    if os.path.exists("reuter_dict"):
        with open("reuter_dict", encoding="utf-8") as f:
            dic = eval(f.read())
        return dic
    all_words=[]
    for c in corpus:
        text=c["text"]
        all_words+=process(text)
    dic={}
    for i,w in enumerate(set(all_words)):
        dic[w]=i
    with open("reuter_dict","w",encoding="utf-8") as f:
        f.write(str(dic))
    return dic

def make_data(corpus,dimension=500):
    if os.path.exists("knn_train_data.npy"):
        train_data=np.load("knn_train_data.npy")
        test_data=np.load("knn_test_data.npy")
        return train_data,test_data

    dic=build_dict(corpus)
    train_data,test_data=[],[]
    topic_code={}

    for c in corpus:
        text,topic=c['text'],c['topic']
        words=process(text)
        vector=[0 for _ in range(len(dic))]
        for w in words:
            vector[dic[w]]+=1
        if topic:
            for t in topic:
                if t not in topic_code:
                    topic_code[t] = len(topic_code)
                train_data.append(vector+[topic_code[t]])
        else:
            test_data.append(vector)

    with open("topic_code","w",encoding="utf-8") as f:
        f.write(str(topic_code))

    train_data,test_data=np.array(train_data,dtype='float16'),np.array(test_data,dtype='float16')

    standardscaler = StandardScaler()
    pca = PCA(n_components=dimension)
    x_train,y_train=train_data[:,:-1],train_data[:,-1]
    x_train= standardscaler.fit_transform(x_train)
    test_data= standardscaler.transform(test_data)
    x_train=pca.fit_transform(x_train)
    test_data=pca.transform(test_data)
    train_data=np.hstack((x_train,y_train.reshape(-1,1)))
    np.save("knn_train_data.npy",train_data)
    np.save("knn_test_data.npy",test_data)
    return train_data,test_data

def knn_train(train_data):
    with open("topic_code",encoding="utf-8") as f:
        topic_code=eval(f.read())
    #x_train,y_train=train_data[:,:-1],train_data[:,-1]
    # class_count=[0 for _ in range(len(topic_code))]
    data_group=[[] for _ in range(len(topic_code))]
    for i in range(len(train_data)):
        data_group[int(train_data[i,-1])].append(train_data[i])
    new_train_data=[]
    for i in range(len(data_group)):
        n=len(data_group[i])
        if n<100:
            for _ in range(100):
                index=randint(0,n-1)
                new_train_data.append(data_group[i][index])
        else:
            for index in sample(range(n),100):
                new_train_data.append(data_group[i][index])
    new_train_data=np.array(new_train_data)
    x_train,y_train=new_train_data[:,:-1],new_train_data[:,-1].astype('int')
    print("training...")
    knn_clf = KNeighborsClassifier(n_neighbors=10)
    knn_clf.fit(x_train, y_train)
    joblib.dump(knn_clf, 'knn.model')

def text_classify(test_data,corpus):
    with open("topic_code",encoding="utf-8") as f:
        topic_code=eval(f.read())
    topics=[k for k in topic_code]
    knn_model=joblib.load('knn.model')
    y_pre=knn_model.predict(test_data)
    index=0
    for c in corpus:
        if not c['topic']:
            c['topic']=[topics[y_pre[index]]]
            index+=1
    topic_text={}
    for c in corpus:
        for t in c['topic']:
            if t not in topic_text:
                topic_text[t]=[]
            topic_text[t].append(c['text'])

    with open("topic_text","w",encoding="utf-8") as f:
        f.write(str(topic_text))


if __name__ == '__main__':
    corpus=get_corpus()
    train_data,test_data=make_data(corpus)
    #knn_train(train_data)
    text_classify(test_data,corpus)

