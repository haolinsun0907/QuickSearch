# QuickSearch
This is a search engine I wrote during Univerisity. It supports searching and ranking on local documents with rich functionalities.

The system developed by python 3.6, so please run the code in python 3.6 or newer. The follwing libraries have to be installed before you run the code:

- beautifulsoup4
- langdetect
- nltk
- PyQt5
- sklearn

To run the app:
--------------

Simply run MainWindow.py, the UI will pop up.

There are several modules to implement different functionalities:

Model 1 & 2 (Bigram model + Query completion)
--------------------
The implementation of the model is the class named “BigramLanguageModel” in BigramLanguageModel.py. The Bigram Language Model class need corpus to create a model. The model’s algorithm computes the probability of next words of a word based on analysis of the corpus words.
The query completion model is implemented in the function query_completion() in Modules.py. The function needs a Bigram Language Model and a query, it returns a suggested completed query list.

### Functionality: 
The main function of these models is that after the user enters a word, the system will recommend the next word that the user may want to enter based on the frequency of occurrence of all words that appear after this word in the database. The recommended words will appear in the form of a drop-down menu. If the user selects a word in the drop-down menu, the word will be automatically added to the search bar.

Model 3 (Query expansion with Wordnet)
-----------------
This module is implemented in the function global_query_expansion() in Modules.py. The function needs a query string, then it will return some words similar to the words of the query sorted by their similarities. 

### Functionality: 
The main function of this model is that after the user searches for one or a few words, the system will automatically recommend the user to some words related to the word searched by the user, so that the user can further narrow the search scope to conduct a more accurate search. 

### Problem encountered:
The problem we encountered in this model is that there are too many synonyms or synonyms for many words, so sometimes it is impossible to give users effective suggestions. In order to deal with this problem, we have set up a mechanism. In the case of too many synonyms, we choose not to display any suggestions. We want to use this method to reduce the possibility of misleading users.

Model 4&5 (Relevance feedback + query expansion with Rocchio)
---------------------
This module is implemented in the function Rocchio_algorithm () in Modules.py. The function needs a query string, relevant documents ids, not relevant documents ids and all documents’ vectors. The algorithm will computer a more reasonable query according the search results’ relevance.

### Functionality: 
The main function of this model is to allow the user to choose whether he is satisfied with the search results. In order to achieve this goal, we have added two options after all returned results: checkmark that represents "relevant", and cross mark represents "irrelevant". After seeing the search results, the user can decide whether the given search results are what he wants. If a search result is what he wants, he can mark the search result as "relevant", so the next time he searches for the same keywords, the related results he marked earlier will be displayed at the top. Similarly, if the user marks a search result as "irrelevant", then this result will not appear at the top next time.

Model 6 (topic classification)
------------------
The implementation of the module can be found in text_categorization.py. Some import functions are explained as following:  
	parse_content: parsing the file content of Reuters corpus and topic and text of each document.  
	build_dict: building the word dictionary of the Reuters corpus.  
	make_data: each document is grouped by their topics. If a document has serval topics, then it will be grouped into several groups.   Every topic assigned as a label to the vector of the document text will be the training data. If a document has no topic, it will be seen as test data later on. The training data and test data are saved after normalization and dimension reduction.  
	knn_train : train the KNN model based on train data and save into files.  
	text_classify: predict a label for each row of the test data using trained KNN model, then translate the label into topic and assign it to the corresponding document. At last save all the documents.  

### Functionality: 
The main function of this model is to give the user some options based on the topics related to the keyword after the user enters the keyword. The user can select one of the topics to view all news with this keyword on this topic. This feature enhances the user's search efficiency and can filter out topics that the user does not want to see. 

### Problem encountered: 
But this function also has some problems. We found that not all news in the database will have a topic attached, and one document can contain multiple topics. This will cause users to miss some news that actually matches the desired topics when searching. The classification seems to make sense base on the result, but maybe not that accurate. 

#### Is kNN a good approach?
We think kNN is a fair approach, but there may have some better ways to deal with this. The reason is that there were numerous topics to deal with, but if we just simply do the recommendation by kNN, we will certainly miss some real valuable information since the justifying standard of kNN is too simple and not comprehensive enough. 

