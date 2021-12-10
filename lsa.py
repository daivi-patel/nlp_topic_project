# https://medium.com/@osas.usen/topic-extraction-from-tweets-using-lda-a997e4eb0985
# https://www.machinelearningplus.com/nlp/topic-modeling-python-sklearn-examples/
import nltk
import gensim
from pprint import pprint
import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import re
import numpy as np
import pandas as pd
import re, nltk, spacy, gensim

#import nltk
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize,sent_tokenize

#preprocessing
from nltk.corpus import stopwords  #stopwords
from nltk import word_tokenize,sent_tokenize # tokenizing
from nltk.stem import PorterStemmer,LancasterStemmer  # using the Porter Stemmer and Lancaster Stemmer and others
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer  # lammatizer from WordNet

# for named entity recognition (NER)
from nltk import ne_chunk

# vectorizers for creating the document-term-matrix (DTM)
from sklearn.feature_extraction.text import TfidfVectorizer

# Sklearn
from sklearn.decomposition import TruncatedSVD
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from pprint import pprint


my_stopwords = nltk.corpus.stopwords.words('english')
word_rooter = nltk.stem.snowball.PorterStemmer(ignore_stopwords=False).stem


def clean_tweet(tweet, bigrams=False):
    tweet_token_list = [word for word in tweet.split(' ') if word not in my_stopwords]  # remove stopwords
    tweet_token_list = [word_rooter(word) if '#' not in word else word for word in
                        tweet_token_list]  # apply word rooter
    if bigrams:
        tweet_token_list = tweet_token_list + [tweet_token_list[i] + '_' +
                                               tweet_token_list[i + 1] for i in range(len(tweet_token_list) - 1)]
    tweet = ' '.join(tweet_token_list)
    return tweet


def sent_to_words(sentences):
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations


def main(df):
    # print("Turning data into list")
    data = df.Text.values.tolist()
    # pprint(data[:1])
    data_words = list(sent_to_words(data))
    # print("Done turning data into list")
    # print(data_words)
    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # Run in terminal: python3 -m spacy download en
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    def lemmatization(texts, allowed_postags=None):
        """https://spacy.io/api/annotation"""
        if allowed_postags is None:
            allowed_postags = ['NOUN', 'ADJ', 'VERB', 'ADV']
        texts_out = []
        for sent in texts:
            # print(sent)
            doc = nlp(" ".join(sent))
            # print(doc)
            texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if
                                       token.pos_ in allowed_postags]))
        return texts_out

    # Do lemmatization keeping only Noun, Adj, Verb, Adverb
    # print("Lemmatizing")
    data_lemmatized = lemmatization(data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    # print("Done lemmatizing")
    # print(data_lemmatized)

    vectorizer = TfidfVectorizer(stop_words='english', lowercase = True, token_pattern='[a-zA-Z0-9]{3,}', max_df = 0.1, min_df = 0.01)
    data_vectorized = vectorizer.fit_transform(data_lemmatized)

    lsa_model = TruncatedSVD(n_components=10, algorithm='randomized', n_iter=12, random_state=42)
    lsa_output = lsa_model.fit_transform(data_vectorized)

    # print(vectorizer.get_feature_names())
    # vocab = vectorizer.get_feature_names()
    #
    # for i, comp in enumerate(lsa_model.components_):
    #     vocab_comp = zip(vocab, comp)
    #     sorted_words = sorted(vocab_comp, key=lambda x: x[1], reverse=True)[:15]
    #     print("Topic " + str(i) + ": ")
    #     for t in sorted_words:
    #         print(t[0] + " ")
    #     print("\n")
    #

    # logreg_lsa = LogisticRegression()
    # logreg_param_grid = [{'penalty': ['l1', 'l2']},
    #                      {'tol': [0.0001, 0.0005, 0.001]}]
    # grid_lsa_log = GridSearchCV(estimator=logreg_lsa,
    #                             param_grid=logreg_param_grid,
    #                             scoring='accuracy', cv=5,
    #                             n_jobs=-1)
    # best_lsa_logreg = grid_lsa_log.fit(lsa_output, data).best_estimator_
    # print("Accuracy of Logistic Regression on LSA train data is :", best_lsa_logreg.score(lsa_output, data))

    # column names
    topicnames = ["Topic" + str(i) for i in range(lsa_model.n_components)]

    # index names
    docnames = ["Doc" + str(i) for i in range(len(data))]
    #
    # # Make the pandas dataframe
    df_document_topic = pd.DataFrame(np.round(lsa_output, 2), columns=topicnames, index=docnames)
    #
    # # Get dominant topic for each document
    dominant_topic = np.argmax(df_document_topic.values, axis=1)
    df_document_topic['dominant_topic'] = dominant_topic
    #
    # Styling
    def color_green(val):
        color = 'green' if val > .1 else 'black'
        return 'color: {col}'.format(col=color)

    def make_bold(val):
        weight = 700 if val > .1 else 400
        return 'font-weight: {weight}'.format(weight=weight)
    # #
    # # Apply Style
    # df_document_topics = df_document_topic.head(15).style.applymap(color_green).applymap(make_bold)
    print(df_document_topic)
    df_topic_distribution = df_document_topic['dominant_topic'].value_counts().reset_index(name="Num Documents")
    df_topic_distribution.columns = ['Topic Num', 'Num Documents']

    # Show top n keywords for each topic
    def show_topics(vectorizer=vectorizer, lsa_model=lsa_model, n_words=20):
        keywords = np.array(vectorizer.get_feature_names())
        topic_keywords = []
        for topic_weights in lsa_model.components_:
            top_keyword_locs = (-topic_weights).argsort()[:n_words]
            topic_keywords.append(keywords.take(top_keyword_locs))
        return topic_keywords

    topic_keywords = show_topics(vectorizer=vectorizer, lsa_model= lsa_model, n_words=15)

    # Topic - Keywords Dataframe
    df_topic_keywords = pd.DataFrame(topic_keywords)
    df_topic_keywords.columns = ['Word ' + str(i) for i in range(df_topic_keywords.shape[1])]
    df_topic_keywords.index = ['Topic ' + str(i) for i in range(df_topic_keywords.shape[0])]
    print(df_topic_keywords)