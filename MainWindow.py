import sys
import json
import random
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn


ps = PorterStemmer()
G_cookies = {}
with open("processed/courses/word_bigrams.json", encoding="utf-8") as f:
    courses_bigrams = json.load(f)
with open("processed/reuters/word_bigrams.json", encoding="utf-8") as f:
    reuters_bigrams = json.load(f)


class DetailDialog(QDialog):
    def __init__(self, title, content, parent):
        super().__init__(parent)
        self.title = title
        self.content = content
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setupUi()

    def setupUi(self):
        self.setObjectName("Form")
        self.setWindowTitle("detail")
        self.setFixedSize(444, 532)
        self.label = QLabel(self)
        self.label.setGeometry(120, 30, 431, 21)
        # self.label.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.label.setObjectName("label")
        self.textBrowser = QTextBrowser(self)
        self.textBrowser.setGeometry(40, 60, 331, 451)
        self.textBrowser.setObjectName("textBrowser")
        self.label.setText(self.title)
        self.textBrowser.setText(self.content)
        self.setWindowModality(Qt.ApplicationModal)

#Inspired from # https://www.geeksforgeeks.org/python-stemming-words-with-nltk/
def stemmer(word):
    return ps.stem(word)


def is_vowel(c):
    return c == 'a' or c == 'e' or c == 'i' or c == 'o' or c == 'u' or c == 'y'


def normalization(text):
    return text.replace('-', '').replace('.', '')


def is_stopword(word):
    return word in stopwords.words("english")


def is_operator(token):
    return token == 'AND' or token == 'OR' or token == 'AND_NOT'


def peek(stack):
    return stack[-1] if stack else None


def infix_to_postfix(query):
    tokens = word_tokenize(query)
    operatorStack = []
    outputQueue = []
    for token in tokens:
        if token == '(':
            operatorStack.append('(')
        elif token == ')':
            top = peek(operatorStack)
            while top is not None and top != '(':
                outputQueue.append(operatorStack.pop())
                if not operatorStack:
                    print('Unbalanced parentheses')
                    return outputQueue
                top = peek(operatorStack)
            operatorStack.pop()
        elif is_operator(token):
            top = peek(operatorStack)
            while top is not None and top not in '()':
                outputQueue.append(operatorStack.pop())
                top = peek(operatorStack)
            operatorStack.append(token)
        else:  # Word
            outputQueue.append(token)

    while operatorStack:
        outputQueue.append(operatorStack.pop())

    return outputQueue

# Inspired from https://stackoverflow.com/questions/5486337/how-to-remove-stop-words-using-nltk-or-python
def clean(word, settings):
    if word == '':
        return None
    if settings['stemming']:
        word = stemmer(word)
    if settings['normalization']:
        word = normalization(word)
    if settings['stopword_removal'] and is_stopword(word):
        word = None
    return word


def boolean_calculate(ids1, ids2, operator):
    answer = []
    if operator == 'AND':
        p1 = 0
        p2 = 0
        while p1 < len(ids1) and p2 < len(ids2):
            if ids1[p1] == ids2[p2]:
                answer.append(ids1[p1])
                p1 += 1
                p2 += 1
            elif ids1[p1] < ids2[p2]:
                p1 += 1
            else:
                p2 += 1
    elif operator == 'OR':
        p1 = 0
        p2 = 0
        while p1 < len(ids1) and p2 < len(ids2):
            if ids1[p1] == ids2[p2]:
                answer.append(ids1[p1])
                p1 += 1
                p2 += 1
            elif ids1[p1] < ids2[p2]:
                answer.append(ids1[p1])
                p1 += 1
            else:
                answer.append(ids2[p2])
                p2 += 1

        while p1 < len(ids1):
            answer.append(ids1[p1])
            p1 += 1
        while p2 < len(ids2):
            answer.append(ids2[p2])
            p2 += 1
    else:
        p1 = 0
        p2 = 0
        while p1 < len(ids1) and p2 < len(ids2):
            if ids1[p1] == ids2[p2]:
                p1 += 1
                p2 += 1
            elif ids1[p1] < ids2[p2]:
                p1 += 1
            else:
                answer.append(ids2[p2])
                p2 += 1
        while p2 < len(ids2):
            answer.append(ids2[p2])
            p2 += 1
    return answer


def word_to_ids(word_list):
    ids = []
    for item in word_list:
        ids.append(item['id'])
    return ids


def handle_wildcard(word, index, processed_path):
    split = word.split('*')
    word_bigram = []
    for part in split:
        if part != '':
            if part == split[0]:
                word_bigram.append('$' + part[0])
            elif part == split[-1]:
                word_bigram.append(part[-1] + '$')

            for i in range(0, len(part) - 1):
                word_bigram.append(part[i:i + 2])

    if len(word_bigram) == 0:
        return []

    with open(processed_path + '/letter_bigrams.json', encoding="utf-8") as file:
        bigrams = json.load(file)
        words = sorted(bigrams[word_bigram[0]])
        for i in range(1, len(word_bigram)):
            words = boolean_calculate(words, sorted(bigrams[word_bigram[i]]), 'AND')

        if len(words) == 0:
            return []

        documents = []
        for word in words:
            documents = boolean_calculate(documents, word_to_ids(index[word]['docs']), 'OR')
        return documents


def weighted_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                weight = 3
                if is_vowel(c1) and is_vowel(c2):
                    weight = 1
                elif not is_vowel(c1) and not is_vowel(c2):
                    weight = 2
                elif (not c1 or not c2) and (is_vowel(c1) or is_vowel(c2)):
                    weight = 1

                distances_.append(weight + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

# Modified from https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
def correction(null_word, processed_path):
    if len(null_word) < 3 or is_stopword(null_word):
        return None

    null_word = null_word.lower()

    possiblilities = []
    with open(processed_path + '/raw_dictionary.json', encoding="utf-8") as file:
        words = json.load(file)
        for word in words:
            if word[0] == null_word[0] and len(
                    word) < len(null_word) + 3 and len(word) > len(null_word) - 3 and not is_stopword(word):
                possiblilities.append(word)

        if not possiblilities:
            return None

        candidate = {}
        for choice in possiblilities:
            edit_distance = weighted_distance(null_word, choice)
            if edit_distance < 5:
                candidate[choice] = words[choice]

        return [key for key, value in sorted(candidate.items(), key=lambda item: item[1], reverse=True)[:3]]


def boolean_search(query, index, settings, processed_path):
    infix = infix_to_postfix(query)
    stack = []
    possible_corrections = {}
    for item in infix:
        if is_operator(item):
            ids1 = stack.pop()
            ids2 = stack.pop()
            stack.append(boolean_calculate(ids1, ids2, item))
        else:
            ids = []
            cleaned = clean(item, settings)
            if '*' in cleaned:
                ids = handle_wildcard(cleaned, index, processed_path)
            elif cleaned and cleaned in index:
                ids = word_to_ids(index[cleaned]['docs'])
            else:
                tmp_correction = correction(item, processed_path)
                if tmp_correction:
                    possible_corrections[item] = tmp_correction

            stack.append(ids)
    return stack.pop(), possible_corrections

#inspired from https://github.com/aimannajjar/columbiau-rocchio-search-query-expander/blob/master/rocchio.py
def rocchio_query_expansion(query_vector, index, settings, collection, relevant_cookie):
    if relevant_cookie:
        relevance_dict = json.loads(relevant_cookie)
        relevant_vector = {}
        irrelevant_vector = {}
        count_relevant = 0
        count_irrelevant = 0
        if collection in relevance_dict:
            for query in relevance_dict[collection]:
                tokens = word_tokenize(query)
                for token in tokens:
                    cleaned = clean(token, settings)
                    if cleaned in index:
                        for item in index[cleaned]['docs']:
                            if str(item['id']) in relevance_dict[collection][query]:
                                if relevance_dict[collection][query][str(item['id'])]:
                                    count_relevant += 1
                                    if cleaned in relevant_vector:
                                        relevant_vector[cleaned] += item['tf']
                                    else:
                                        relevant_vector[cleaned] = item['tf']
                                else:
                                    count_irrelevant += 1
                                    if cleaned in irrelevant_vector:
                                        irrelevant_vector[cleaned] += item['tf']
                                    else:
                                        irrelevant_vector[cleaned] = item['tf']
        for word in relevant_vector:
            value = 0.75 * (relevant_vector[word] / count_relevant)
            if word in query_vector:
                query_vector[word] += value
            else:
                query_vector[word] = value

        for word in irrelevant_vector:
            value = -0.15 * (irrelevant_vector[word] / count_irrelevant)
            if word in query_vector:
                query_vector[word] += value
            else:
                query_vector[word] = value
    return query_vector

# Modified from https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
def vsm_search(query, index, settings, processed_path, collection, relevant_cookie):
    tokens = word_tokenize(query)
    query_vector = {}
    possible_corrections = {}

    for token in tokens:
        tmp = clean(token, settings)
        if tmp not in index:
            tmp_correction = correction(token, processed_path)
            if tmp_correction:
                possible_corrections[token] = tmp_correction
        if tmp in query_vector:
            query_vector[tmp] += 1
        else:
            query_vector[tmp] = 1

    query_vector = rocchio_query_expansion(query_vector, index, settings, collection, relevant_cookie)

    docs_vector = {}
    for word in query_vector:
        if word in index:
            for doc in index[word]['docs']:
                value = query_vector[word] * doc['tf'] * index[word]['idf']
                if doc['id'] in docs_vector:
                    docs_vector[doc['id']] += value
                else:
                    docs_vector[doc['id']] = value
    return ([(key, value) for key, value in sorted(docs_vector.items(), key=lambda item: item[1], reverse=True)[:15]],
            possible_corrections, query_vector)


def search(query, model, processed_path, collection, relevant_cookie):
    with open(processed_path + '/index.json', encoding="utf-8") as indexFile:
        with open('processed/settings.json', encoding="utf-8") as settingsFile:
            index = json.load(indexFile)
            settings = json.load(settingsFile)
            if model == 'boolean':
                return boolean_search(query, index, settings, processed_path)
            else:
                return vsm_search(query, index, settings, processed_path, collection, relevant_cookie)

# Inspired from: https://www.nltk.org/howto/wordnet.html
def search_results(request):
    collection = request.get('collection')
    processed_path = 'processed/' + collection
    query = request.get('query')
    results = search(query, request.get('model'), processed_path, collection, request.get('relevant'))
    documents = {}
    topics = {}
    with open(processed_path + '/preprocessed.json', encoding="utf-8") as file:
        corpus = json.load(file)
        for doc in results[0]:
            if request.get('model') == 'boolean':
                documents[doc] = (corpus[doc], None)
            elif request.get('model') == 'vsm':
                documents[doc[0]] = (corpus[doc[0]], doc[1])
    expansion = {}
    for word in word_tokenize(query):
        synsets = wn.synsets(word)
        if synsets and len(synsets) < 10 and word not in expansion:
            expansion[word] = []
            for synset in wn.synsets(word):
                for hypernym in synset.hypernyms():
                    for name in hypernym.lemma_names():
                        if '_' not in name and name not in expansion[word] and name not in query:
                            expansion[word].append(name)
            if len(expansion[word]) == 0:
                del expansion[word]
    if collection == 'reuters':
        with open(processed_path + '/topics.json', encoding="utf-8") as file:
            tmp = json.load(file)
            for doc in documents:
                topic = tmp[doc]
                if len(topic['topics']) == 0:
                    if 'none' in topics:
                        topics['none'].append(doc)
                    else:
                        topics['none'] = [doc]
                for t in topic['topics']:
                    if t in topics:
                        topics[t].append(doc)
                    else:
                        topics[t] = [doc]
    vsm_score = None
    if len(results) > 2:
        vsm_score = results[2]

    context = {
        'collection': collection,
        'query': query,
        'documents': documents,
        'corrections': results[1],
        'topics': topics,
        'expansion': expansion,
        'vsm_score': vsm_score
    }
    with open('res.txt', 'w', encoding="utf-8") as f:
        f.write(json.dumps(context, ensure_ascii=False))
    return context


def document(collection, doc_id):
    with open('processed/' + collection + '/preprocessed.json', encoding="utf-8") as file:
        corpus = json.load(file)
        context = {'doc_id': doc_id, 'document': corpus[doc_id]}
        if collection == 'reuters':
            with open('processed/' + collection + '/topics.json', encoding="utf-8") as topicsFile:
                context['topic'] = json.load(topicsFile)[doc_id]
        return context


class Worker(QThread):
    sinOut = pyqtSignal(dict)

    def __init__(self, parent=None):
        super(Worker, self).__init__(parent)
        self.data = {}

    def run(self):
        if self.data:
            res = search_results(self.data)
            self.sinOut.emit(res)


class Worker2(QThread):
    sinOut = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = {}

    def run(self):
        if self.data:
            res = document(self.data.get('collection'), self.data.get("doc_id"))
            self.sinOut.emit(res)


class BaseWidget(QWidget):
    def __init__(self, parent=None):
        super(BaseWidget, self).__init__(parent)
        self.setStyleSheet("QWidget{background:transparent;}")
        self.setMouseTracking(True)

    def paintEvent(self, a0: QPaintEvent):
        opt = QStyleOption()
        opt.initFrom(self)
        p = QPainter(self)
        self.style().drawPrimitive(QStyle.PE_Widget, opt, p, self)

    def mousePressEvent(self, a0: QMouseEvent):
        super(BaseWidget, self).mousePressEvent(a0)

    def mouseMoveEvent(self, a0: QMouseEvent):
        super(BaseWidget, self).mouseMoveEvent(a0)


class Main(QWidget):
    def __init__(self, parent=None):
        super(Main, self).__init__(parent)
        self.expansion = []
        self.resultData = {}
        self.currentindex = 0
        self.radioindex = 3
        self.child = None
        self.setupUi()
        self.work = Worker()
        self.work.sinOut.connect(self.result)
        self.work2 = Worker2()
        self.work2.sinOut.connect(self.result2)
# Modified from https://blog.csdn.net/weixin_39449466/article/details/81008711
    def setupUi(self):
        self.setObjectName("Form")
        self.resize(1024, 768)
        font = QFont()
        font.setFamily("Arial Black")
        font.setUnderline(True)
        font.setKerning(True)
        self.mainlayout = QVBoxLayout()
        self.hlayout_1 = QHBoxLayout()
        self.hlayout_2 = QHBoxLayout()
        self.lineEdit = QLineEdit()
        self.lineEdit.setFixedSize(300, 30)
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit.setPlaceholderText("Quick Search Now:")
        self.pushButton = QPushButton()
        self.pushButton.setFixedSize(70, 30)
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setText("Search")
        self.combox = QComboBox()
        self.combox.setFixedSize(150, 30)
        self.combox.addItem("Choose topic")
        self.widget = BaseWidget()
        self.widget.setFixedSize(360, 600)
        self.widget.setProperty("name", 'w')
        self.widget_1 = BaseWidget(self.widget)
        self.widget_1.setGeometry(0, 0, 360, 100)
        self.widget_1.setStyleSheet("background-color: white;")


        self.listWidget = QListWidget(self.widget)
        self.listWidget_2 = QListWidget(self.widget)
        self.listWidget.setGeometry(0, 100, 180, 500)
        self.listWidget.setProperty("name", 'l')
        self.listWidget_2.setProperty("name", 'l')
        self.listWidget_2.setGeometry(180, 100, 180, 500)

        self.label = QLabel(self.widget_1)
        self.label.setText("Wordnet Expansion")
        self.label4= QLabel(self.widget_1)
        self.label.setGeometry(40, 20, 250, 40)
        self.label4.setGeometry(20, 0, 200, 20)
        self.label2 = QLabel(self.widget_1)
        self.label3 = QLabel(self.widget_1)
        self.label2.setGeometry(40, 60, 100, 40)
        self.label3.setGeometry(180, 60, 100, 40)
        self.expansion.append([self.label2, self.listWidget])
        self.expansion.append([self.label3, self.listWidget_2])
        font.setPixelSize(20)
        self.label.setFont(font)
        self.textBrowser = QTextBrowser()
        self.textBrowser.setFixedSize(600, 600)
        self.textBrowser.setObjectName("textBrowser")
        self.groupBox = QGroupBox()
        self.groupBox.setFixedSize(90, 80)
        self.groupBox.setObjectName("groupBox")
        self.groupBox.setTitle("Search Model")
        self.radioButton_2 = QRadioButton(self.groupBox)
        self.radioButton_2.setGeometry(QRect(10, 20, 90, 16))
        self.radioButton_2.setObjectName("radioButton_2")
        self.radioButton = QRadioButton(self.groupBox)
        self.radioButton.setGeometry(QRect(10, 50, 90, 16))
        self.radioButton.setObjectName("radioButton")
        self.groupBox_2 = QGroupBox()
        self.groupBox_2.setFixedSize(90, 80)
        self.groupBox_2.setObjectName("groupBox_2")
        self.groupBox_2.setTitle("Collection")
        self.radioButton_3 = QRadioButton(self.groupBox_2)
        self.radioButton_3.setGeometry(QRect(0, 20, 90, 16))
        self.radioButton_3.setObjectName("radioButton_3")
        self.radioButton_4 = QRadioButton(self.groupBox_2)
        self.radioButton_4.setGeometry(QRect(0, 50, 90, 16))
        self.radioButton_4.setObjectName("radioButton_4")

        self.radioButton_2.setText("Boolean")
        self.radioButton.setText("VSM")
        self.radioButton_3.setText("uOttawa")
        self.radioButton_4.setText("Reuters")

        self.hlayout_1.addWidget(self.groupBox)
        self.hlayout_1.addWidget(self.groupBox_2)
        self.hlayout_1.addWidget(self.lineEdit)
        self.hlayout_1.addWidget(self.pushButton)
        self.hlayout_1.addWidget(self.combox)
        self.hlayout_2.addWidget(self.textBrowser)
        self.hlayout_2.addWidget(self.widget)
        self.mainlayout.addLayout(self.hlayout_1)
        self.mainlayout.addLayout(self.hlayout_2)

        self.radioButton_2.setChecked(True)
        self.radioButton_3.setChecked(True)
        self.textBrowser.setOpenLinks(False)
        self.textBrowser.anchorClicked.connect(self.show_content)
        self.pushButton.clicked.connect(self.search)
        self.combox.activated.connect(self.comboxchange)
        self.listWidget.itemClicked.connect(self.listitemclicked)
        self.listWidget_2.itemClicked.connect(self.listitemclicked)
        self.radioButton_3.clicked.connect(self.radioButton_34clicked)
        self.radioButton_4.clicked.connect(self.radioButton_34clicked)
        self.lineEdit.textChanged.connect(self.lineeditChangedsignal)
        self.setLayout(self.mainlayout)
        self.completer = QCompleter()
# Inspired from https://blog.csdn.net/chengmo123/article/details/93379470
        self.lineEdit.setCompleter(self.completer)
        self.model = QStringListModel()
        self.completer.setModel(self.model)
        self.get_data()

    def lineeditChangedsignal(self):
        text = self.lineEdit.text()
        if text.strip():
            if text.endswith(" "):
                new_text = text.strip().split(" ")[-1]
                if self.radioindex == 3:
                    words = courses_bigrams.get(new_text)
                    if words:
                        for i in words:
                            word = text.strip() + " " + i
                            self.get_data(add=1, word=word)
                else:
                    words = reuters_bigrams.get(new_text)
                    if words:
                        for i in words:
                            word = text.strip() + " " + i
                            self.get_data(add=1, word=word)

    def radioButton_34clicked(self):
        if self.radioButton_3.isChecked():
            if self.radioindex != 3:
                self.radioindex = 3
                self.get_data()
        else:
            if self.radioindex != 4:
                self.radioindex = 4
                self.get_data(index=0)

    def get_data(self, index=1, add=0, word=""):
        if add:
            self.model.insertRow(self.model.rowCount())
            currentindex = self.model.index(self.model.rowCount()-1, 0)
            self.model.setData(currentindex, word)
        else:
            if index:
                self.model.setStringList([i for i in courses_bigrams])
            else:
                self.model.setStringList([i for i in reuters_bigrams])

    def show_content(self, url: QUrl):
        txt = url.url()
        tmp, txt = txt.split(":", 1)
        if tmp == "relevant":
            q, i, b = txt.split(":")
            if int(b):
                b = True
            else:
                b = False
            if G_cookies.get('reuters'):
                if G_cookies['reuters'].get(q):
                    G_cookies['reuters'][q][i] = b
                else:
                    G_cookies['reuters'][q] = {i: b}
            else:
                G_cookies['reuters'] = {q: {i: b}}
        elif tmp == "detail":
            r, i = txt.split(":", 1)
            self.work2.data = {"collection": r, "doc_id": int(i)}
            self.work2.start()

    def comboxchange(self):
        if self.combox.currentIndex() != self.currentindex:
            self.currentindex = self.combox.currentIndex()
            if self.currentindex == 0:
                self.setresult()
            else:
                self.setresult(self.resultData['topics'][self.combox.currentText()])

    def result(self, d):
        self.combox.clear()
        self.combox.addItem("Choose topic")
        self.pushButton.setEnabled(True)
        self.resultData = d
        res_cout = len(self.resultData['documents'])
        self.label4.setText("{}({})".format(self.resultData.get("query"), res_cout))
        c = 0
        for i in self.resultData['topics']:
            self.combox.addItem(i)
        for x, y in self.resultData['expansion'].items():
            self.expansion[c][0].setText("<font color='black'><b> {} </b> </font>".format(x))
            random.shuffle(y)
            self.expansion[c][1].addItems(y)
            c = (c + 1) % 2
        self.setresult()

    def setresult(self, mode="all"):
        self.textBrowser.document().clear()
        collection = self.resultData['collection']
        query = self.resultData['query']
        for x, y in self.resultData['documents'].items():
            if mode == "all" or x in mode:
                title = y[0].get('title')
                excerpt = y[0].get('body')[:100]
                link = str(x)
                self.textBrowser.append("<a href={}:{}:{}>{}</a>\n".format("detail", collection, link, title))
                self.textBrowser.append(excerpt + "\n")
                if collection == "reuters":
                    self.textBrowser.append(
                        "<a href={}:{}:{}:{}>{}</a>  <a href={}:{}:{}:{}>{}</a>\n".format(
                            "relevant", query, x, 1, "√", "relevant", query, x, 0, "×"))
                self.textBrowser.append("-" * 75 + "\n")

        self.textBrowser.moveCursor(self.textBrowser.textCursor().Start)

    def result2(self, d):
        self.child = DetailDialog(d['document']['title'], d['document']['body'], self)
        self.child.show()

    def listitemclicked(self, item):
        text = self.lineEdit.text().strip().split()
        if item.text() not in text:
            self.lineEdit.setText(self.lineEdit.text().strip() + " " + item.text())
            self.search()
# Inspired from http://blog.sina.com.cn/s/blog_71a803cb0102yywy.html
    def search(self):
        if self.lineEdit.text().strip():
            self.currentindex = 0
            self.textBrowser.document().clear()
            self.pushButton.setEnabled(False)
            self.label4.setText("")
            text = self.lineEdit.text().strip().replace(
                " and ", " AND ").replace(" or ", " OR ").replace(" and_not ", " AND_NOT ")
            for i in self.expansion:
                i[0].clear()
                i[1].clear()
            if self.radioButton_2.isChecked():
                model = "boolean"
            else:
                model = "vsm"
            if self.radioButton_3.isChecked():
                collection = "courses"
            else:
                collection = "reuters"
            d = {'query': text, 'action': "", "model": model, "collection": collection, 'relevant': json.dumps(G_cookies)}
            self.work.data = d
            self.work.start()
        else:
            QMessageBox(self, "info", "Query parameters cannot be empty", QMessageBox.Ok, QMessageBox.Ok)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet("""
    QWidget[name = "w"]{border: 1px solid rgb(224, 224, 224);}
    QListWidget[name = "l"]{border: 0px; background-color: rgb(240,240,240);padding:20px;} 
    """)
    MainWindow = Main()
    MainWindow.show()
    sys.exit(app.exec_())
