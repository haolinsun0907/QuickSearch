from bs4 import BeautifulSoup
from langdetect import detect
import os
from nltk import PorterStemmer
import nltk
from BooleanModel import BooleanModel
from VectorSpaceModel import VectorSpaceModel
from BigramLanguageModel import BigramLanguageModel
from nltk.corpus import wordnet as wn
import numpy as np
import string

# Module 2
def corpus_pre_processing(text):
    if os.path.exists("corpus"):
        with open("corpus",encoding="utf-8") as f:
            corpus=eval(f.read())
        return corpus
    bs = BeautifulSoup(text, "html.parser")
    courses=bs.find_all(class_="courseblock")
    corpus={}
    docid=1
    for i in range(len(courses)):
        title=courses[i].find(class_="courseblocktitle").text.strip()
        lang=detect(title)
        if lang!='en':
            continue
        desc=courses[i].find(class_="courseblockdesc")
        if desc:
            desc=courses[i].find(class_="courseblockdesc").text.strip()
        else:
            desc=""
        corpus[str(docid)]=(title,desc)
        docid+=1
    with open("corpus","w",encoding='utf-8') as f:
        f.write(str(corpus))
    return corpus


def load_stopwords():
    stopwords=[]
    with open("stopwords.txt",encoding='utf-8') as f:
        for line in f.readlines():
            stopwords.append(line.strip())
    return set(stopwords)


def stopword_removal_f(words,stopwords):
    return [w for w in words if w not in stopwords]


def stemming_f(words,stemmer):
    return [stemmer.stem(w) for w in words]


def normalization_f(words):
    result=[]
    for w in words:
        if w in string.punctuation:
            continue
        w=w.replace("-"," ")
        w=w.replace(".","")
        ws=w.split()
        for j in ws:
            result.append(j)
    return result


# Module 3
def dictionary_building(courpus,stopword_removal=True,stemming=False,normalization=True):
    all_words=[]
    if stopword_removal:
        stopwords=load_stopwords()
    if stemming:
        stemmer=PorterStemmer()
    for doc in courpus:
        title=courpus[doc][0].lower()
        desc=courpus[doc][1].lower()
        words=nltk.word_tokenize(title)+nltk.word_tokenize(desc)
        if normalization:
            words=normalization_f(words)
        if stopword_removal:
            words=stopword_removal_f(words,stopwords)
        if stemming:
            words=stemming_f(words,stemmer)
        all_words+=words
    return set(all_words)


# Module 4
def inverted_index_construction(corpus,dict):
    if os.path.exists("td_index"):
        with open("td_index", encoding="utf-8") as f:
            td_index = eval(f.read())
        return td_index
    td_index={}
    for w in dict:
        td_index[w]=set()
        for docid in corpus:
            if w in corpus[docid][0].lower() or w in corpus[docid][1].lower():
                td_index[w].add(docid)
    with open("td_index","w",encoding="utf-8") as f:
        f.write(str(td_index))
    return td_index

# Module 7
# inspired from https://www.cnblogs.com/4everlove/p/3678414.html
def creat_bigram_index(td_index):
    if os.path.exists("bigram_index"):
        with open("bigram_index", encoding="utf-8") as f:
            bigram_index = eval(f.read())
        return bigram_index
    bigram_index={}
    for term in td_index:
        term="$"+term
        for i in range(len(term)-1):
            bigram=term[i:i+2]
            if bigram not in bigram_index:
                bigram_index[bigram]={term[1:]}
            else:
                bigram_index[bigram].add(term[1:])
    with open("bigram_index", "w", encoding="utf-8") as f:
        f.write(str(bigram_index))
    return bigram_index

# Module 5
def corpus_access(corpus,docids):
    if not os.path.exists("./cache/"):
        os.makedirs("./cache")
    data=[]
    for id in docids:
        title=corpus[id][0]
        desc=corpus[id][1]
        excerpt=desc[:desc.find(".")+1]
        link="./cache/"+id
        if not os.path.exists(link):
            with open(link,"w",encoding="utf-8") as f:
                f.write(str({"title":title,"desc":desc}))
        data.append((title,excerpt,link))
    return data

# Module 9
# Modified from https://gist.github.com/JackonYang/8310da13df2b427cb38b90dab6a35d2d
def cal_mindistance(word1, word2):
    if not word1:
        return len(word2 or '') or 0

    if not word2:
        return len(word1 or '') or 0

    size1 = len(word1)
    size2 = len(word2)

    last = 0
    tmp = list(range(size2 + 1))
    value = None

    for i in range(size1):
        tmp[0] = i + 1
        last = i
        for j in range(size2):
            if word1[i] == word2[j]:
                value = last
            else:
                value = 1 + min(last, tmp[j], tmp[j + 1])
            last = tmp[j+1]
            tmp[j+1] = value
    return value

def spelling_correction(word,td_index):
    words_distance={}
    min_d=None
    for term in td_index:
        if word[0]==term[0]:
            d=cal_mindistance(word, term)
            words_distance[term]=d
            if min_d is None:
                min_d=d
            else:
                if d<min_d:
                    min_d=d
    candidate_words=[]
    for w in words_distance:
        if words_distance[w]==min_d:
            candidate_words.append(w)
    if len(candidate_words)==1:
        return candidate_words[0]

    result=None
    frequency=0
    for w in candidate_words:
        f=len(td_index[w])
        if f>frequency:
            frequency=f
            result=w
    return result

# Query Completion Module
def query_completion(bigram_model,query):
    query=query.strip()
    if len(query)==0:
        return []
    word=query.split()[-1]
    bigram_model.cal_bp()

    next_words=[t[0] for t in list(sorted(bigram_model.bp[word].items(),key=lambda x:x[1],reverse=True))]
    if "<s>" in next_words:
        next_words.remove("<s>")
    if "</s>" in next_words:
        next_words.remove("</s>")
    return next_words

# Global Query Expansion Module
def global_query_expansion(query):
    words=[]
    expand_words=[]
    query_words=query.split()
    for w in query_words:
        synsets = wn.synsets(w)
        for ws in synsets:
            #print(ws)
            if ws.pos() in ['n','v']:
                similary_words={}
                #print(ws.definition())
                for wi in ws.definition().split():
                    sw=wn.synsets(wi)
                    if sw:
                        sw=sw[0]
                    else:
                        continue
                    simi=ws.path_similarity(sw)
                    if simi:
                        similary_words[sw]=simi
                expand_words+=[t[0].name() for t in list(sorted(similary_words.items(), key=lambda x: x[1], reverse=True))[:3]]
            if ws.pos() in ['a','s','r']:
                similarity_sets=ws.similar_tos()
                expand_words+=[t.name() for t in similarity_sets[:3]]
    for w in expand_words:
        s=w.split(".")[0]
        if s not in words and s not in query_words:
            s=s.replace("_"," ")
            words.append(s)
    return words

# Local Query Expansion with Rocchio Algorithm

def create_doc_vector(corpus,stopword_removal=True,stemming=False,normalization=True):
    if os.path.exists("doc_vector"):
        with open("doc_vector", encoding="utf-8") as f:
            doc_vector = eval(f.read())
        return doc_vector
    word_dict={}
    with open("td_index", encoding="utf-8") as f:
        td_index = eval(f.read())
    for i,w in enumerate(td_index):
        word_dict[w]=i
    if stopword_removal:
        stopwords=load_stopwords()
    if stemming:
        stemmer=PorterStemmer()
    doc_vector={}
    for doc in corpus:
        vector=[0 for _ in range(len(word_dict))]
        title=corpus[doc][0].lower()
        desc=corpus[doc][1].lower()
        words=nltk.word_tokenize(title)+nltk.word_tokenize(desc)
        if normalization:
            words=normalization_f(words)
        if stopword_removal:
            words=stopword_removal_f(words,stopwords)
        if stemming:
            words=stemming_f(words,stemmer)
        for w in words:
            vector[word_dict[w]]+=1
        doc_vector[doc]=vector
    with open("doc_vector","w",encoding="utf-8") as f:
        f.write(str(doc_vector))
    return doc_vector

def Rocchio_algorithm(query,rdoc,ndoc,doc_vector):
    query=query.split()
    word_dict = {}
    words=[]
    with open("td_index", encoding="utf-8") as f:
        td_index = eval(f.read())
    for i, w in enumerate(td_index):
        words.append(w)
        word_dict[w] = i
    query_v=[0 for _ in range(len(word_dict))]
    for w in query:
        if w in word_dict:
            query_v[word_dict[w]]+=1
    k1,k2=1,0.5
    rdoc_v=np.zeros((1,len(word_dict)))[0]
    ndoc_v=np.zeros((1,len(word_dict)))[0]
    for docid in rdoc:
        rdoc_v+=np.array(doc_vector[docid])
    for docid in ndoc:
        ndoc_v += np.array(doc_vector[docid])
    new_query_v=query_v+k1/len(rdoc)*rdoc_v-k2/len(ndoc)*ndoc_v
    new_query=[]
    for i,n in enumerate(new_query_v):
        if n>0:
            for _ in range(int(round(n))):
                new_query.append(words[i])
    return " ".join(new_query)



if __name__ == '__main__':
    with open("UofO_Courses.html",encoding="utf-8") as f:
        html=f.read()
    corpus=corpus_pre_processing(html)
    # dict=dictionary_building(corpus)
    # td_index=inverted_index_construction(corpus, dict)
    # docids=['1','2','3']
    # r=corpus_access(corpus, docids)

    # bigram_index=creat_bigram_index(td_index)
    # query="man*"
    # vsm=BooleanModel(corpus,td_index,bigram_index)
    # r=vsm.retrieval(query)
    # print(r)
    # # r=spelling_correction("operotion",td_index)
    # # print(r)
    # query="computers graphical"
    # vsm=VectorSpaceModel(corpus,td_index)
    # r=vsm.retrieval(query)
    # print(r)

    # bigram_model=BigramLanguageModel(corpus)
    # query="nature"
    # r=query_completion(bigram_model,query)
    # print(r)
    query="business"
    r=global_query_expansion(query)
    print(r)
    # doc_vector=create_doc_vector(corpus, stopword_removal=True, stemming=False, normalization=True)
    # query="business"
    # rdoc=['150', '17', '42', '109', '148']
    # ndoc=['80', '22', '153', '2']
    # new_q=Rocchio_algorithm(query, rdoc, ndoc,doc_vector)
    # print(new_q)


