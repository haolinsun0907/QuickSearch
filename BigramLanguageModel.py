import os

class BigramLanguageModel:
    def __init__(self,corpus):
        self.corpus=corpus
        self.words=[]
        self.bp={}

    def pre_process(self):
        for docid in self.corpus:
            t,c=self.corpus[docid]
            t=t.lower()
            c=c.lower()
            s=t+" "+ c
            s = s.replace(",", " ")
            s = s.replace("(", ". ")
            s = s.replace(")", " .")
            if s[-1]==".":
                s=s[:-1]
            s="<s> "+s+" </s>"
            s=s.replace("."," </s> <s> ")
            words=s.split()
            self.words+=words

    def cal_bp(self):
        if os.path.exists("bigram_probability"):
            with open("bigram_probability",encoding="utf-8") as f:
                self.bp=eval(f.read())
            return
        self.pre_process()

        count={}
        for w in self.words:
            if w not in count:
                count[w]=0
            count[w]+=1
        bigram_count={}
        for i in range(len(self.words)-1):
            if (self.words[i],self.words[i+1]) not in bigram_count:
                bigram_count[(self.words[i],self.words[i+1])]=0
            bigram_count[(self.words[i], self.words[i + 1])] += 1

        words_set=set(self.words)
        for w in words_set:
            self.bp[w]={}
            for wj in words_set:
                if w==wj:
                    continue
                if (w,wj) in bigram_count:
                    self.bp[w][wj]=bigram_count[(w,wj)]/count[w]
        with open("bigram_probability","w",encoding="utf-8") as f:
            f.write(str(self.bp))

