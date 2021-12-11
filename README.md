# nlp_topic_project
NLP class project performing topic analysis on tweets
Packages to install: re, pandas, nltk, gensim, matplotlib, numpy, spacy, (in sklearn) LatentDirichletAllocation, CountVectorizer, GridSearchCV, ne_chunk
Directory structure:
  main.py file is a "home base" for most of the code. It cleans the data and then calls other methods. Running python main.py will call the summary statistics, noun determination, lda, and lsa methods automatically one by one. All of these files are in the same directory as main. You can comment out methods as you wish to see only one run.
  
  the folder nmf comtains a jupyter notebook that contains headers describing the 4 model variations. It can simply be run by executing each cell sequentially. Make sure to move the test dataset to a folder called 'data' in the same directory as nmf before running cells. The test data can be found in the link below.
  
  There is a data folder that will contain the csv files of data. They are too big to upload to github, so you will need to go to http://help.sentiment140.com/for-students and download the files to put in the data folder to run our code. 
