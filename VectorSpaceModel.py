import nltk
import math
import os


class VectorSpaceModel:

    def __init__(self,corpus,td_index):
        self.corpus=corpus
        self.td_index=td_index
        self.weight_index=self.cal_weight()

    def load_stopwords(self):
        stopwords = []
        with open("stopwords.txt", encoding='utf-8') as f:
            for line in f.readlines():
                stopwords.append(line.strip())
        return set(stopwords)

    def stopword_removal_f(self,words, stopwords):
        return [w for w in words if w not in stopwords]

    def normalization_f(self,words):
        result = []
        for w in words:
            w = w.replace("-", " ")
            w = w.replace(".", "")
            ws = w.split()
            for j in ws:
                result.append(j)
        return result

    def process_corpus(self):
        result={}
        stopwords = self.load_stopwords()
        for doc in self.corpus:
            title = self.corpus[doc][0].lower()
            desc = self.corpus[doc][1].lower()
            words = nltk.word_tokenize(title) + nltk.word_tokenize(desc)
            words = self.stopword_removal_f(words, stopwords)
            words = self.normalization_f(words)
            result[doc]=words
        return result

    def cal_weight(self):
        if os.path.exists("weight_index"):
            with open("weight_index", encoding="utf-8") as f:
                weight_index = eval(f.read())
            return weight_index
        doc_words=self.process_corpus()
        n=len(doc_words)
        weight_index={}
        for term in self.td_index:
            m=len(self.td_index[term])
            weight_index[term]={}
            for docid in self.td_index[term]:
                tf=doc_words[docid].count(term)/len(doc_words[docid])
                idf=math.log(n/m)
                weight_index[term][docid]=tf*idf
        with open("weight_index", "w", encoding="utf-8") as f:
            f.write(str(weight_index))
        return weight_index

    def retrieval(self, query,k=20):
        query_words=query.lower().split()
        doc_score={}
        for docid in self.corpus:
            doc_score[docid]=0
            for w in query_words:
                if w in self.weight_index and docid in self.weight_index[w]:
                    doc_score[docid]+=self.weight_index[w][docid]
        d_order = sorted(doc_score.items(), key=lambda x: x[1], reverse=True)
        if d_order[0][1]==0:
            return []
        return [d[0] for d in d_order][:k]




