
class BooleanModel:
    def __init__(self,corpus,td_index,bigram_index):
        self.corpus=corpus
        self.td_index=td_index
        self.bigram_index = bigram_index
        self.all_doc=self.get_all_docs()

    def get_all_docs(self):
        return set(self.corpus.keys())

    # modified from https://blog.csdn.net/qq_29681777/article/details/83448881
    def query_process(self,query):
        q=query.replace("(","( ")
        q=q.replace(")"," )")
        query_list=q.split()
        priority = {"(": 0, "OR": 1, "AND": 2, "AND_NOT": 2}
        operators = ['(',')','AND','OR','AND_NOT']
        st=[]
        exp = []
        for x in query_list:
            if x not in operators:
                exp.append(x.lower())
            else:
                if len(st)==0 or st[-1] == "(" or x == "(":
                    st.append(x)
                elif x == ")":
                    while st[-1] != "(":
                        exp.append(st.pop())
                    st.pop()
                elif priority[x] <= priority[st[-1]]:
                    exp.append(st.pop())
                    st.append(x)
                else:
                    st.append(x)
        while not len(st)==0:
            exp.append(st.pop())
        return exp

    def get_docs(self,term):
        result=set()
        if "*" in term:
            terms=None
            term="$"+term
            index=term.find("*")
            term=term[index+1:]+term[:index]
            for i in range(len(term)-1):
                b=term[i:i+2]
                if b in self.bigram_index:
                    if terms is None:
                        terms=self.bigram_index[b]
                    else:
                        terms = terms & self.bigram_index[b]
            for t in terms:
                if t in self.td_index:
                    result=result | self.td_index[t]
        else:
            if term in self.td_index:
                result=self.td_index[term]
        return result

    # modified from https://blog.csdn.net/qq_29681777/article/details/83448650
    def retrieval(self,query):
        query_exp=self.query_process(query)
        operators = ['AND','OR','AND_NOT']
        st=[]
        for x in query_exp:
            if x not in operators:
                st.append(x)
                continue
            a=st.pop()
            b=st.pop()
            if isinstance(a,str):
                a = self.get_docs(a)
            if isinstance(b,str):
                b = self.get_docs(b)

            c=set()
            if x == "AND":
                c = b & a
            if x == "OR":
                c = b | a
            if x == "AND_NOT":
                c = b & a
                c=self.all_doc-c
            st.append(c)
        if isinstance(st[-1],str):
            return self.get_docs(st[-1])
        return st.pop()






