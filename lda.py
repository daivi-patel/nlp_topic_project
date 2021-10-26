# https://medium.com/@osas.usen/topic-extraction-from-tweets-using-lda-a997e4eb0985
# https://www.machinelearningplus.com/nlp/topic-modeling-python-sklearn-examples/
import nltk
import gensim
from pprint import pprint
import spacy
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import re
import numpy as np
import pandas as pd
import re, nltk, spacy, gensim

# Sklearn
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint

# Plotting tools
# import pyLDAvis
# import pyLDAvis.sklearn
# import matplotlib.pyplot as plt
# %matplotlib inline

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
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations



def main(df):
    data = df.Text.values.tolist()
    # pprint(data[:1])
    data_words = list(sent_to_words(data))
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
            print(sent)
            doc = nlp(" ".join(sent))
            print(doc)
            texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if
                                       token.pos_ in allowed_postags]))
        return texts_out

    # Do lemmatization keeping only Noun, Adj, Verb, Adverb
    data_lemmatized = lemmatization(data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    print(data_lemmatized)

    vectorizer = CountVectorizer(analyzer='word',
                                 min_df=10,  # minimum reqd occurences of a word
                                 stop_words='english',  # remove stop words
                                 lowercase=True,  # convert all words to lowercase
                                 token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                                 # max_features=50000,             # max number of uniq words
                                 )

    data_vectorized = vectorizer.fit_transform(data_lemmatized)

    # Materialize the sparse data
    data_dense = data_vectorized.todense()

    # Compute Sparcity = Percentage of Non-Zero cells
    print("Sparcity: ", ((data_dense > 0).sum() / data_dense.size) * 100, "%")

    # Build LDA Model
    lda_model = LatentDirichletAllocation(n_components=20,  # Number of topics
                                          max_iter=10,  # Max learning iterations
                                          learning_method='online',
                                          random_state=100,  # Random state
                                          batch_size=128,  # n docs in each learning iter
                                          evaluate_every=-1,  # compute perplexity every n iters, default: Don't
                                          n_jobs=-1,  # Use all available CPUs
                                          )
    lda_output = lda_model.fit_transform(data_vectorized)

    print(lda_model)  # Model attributes

    # Log Likelyhood: Higher the better
    print("Log Likelihood: ", lda_model.score(data_vectorized))

    # Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
    print("Perplexity: ", lda_model.perplexity(data_vectorized))

    # See model parameters
    pprint(lda_model.get_params())
    # Create Document - Topic Matrix
    lda_output = lda_model.transform(data_vectorized)

    # column names
    topicnames = ["Topic" + str(i) for i in range(lda_model.n_components)]

    # index names
    docnames = ["Doc" + str(i) for i in range(len(data))]

    # Make the pandas dataframe
    df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)

    # Get dominant topic for each document
    dominant_topic = np.argmax(df_document_topic.values, axis=1)
    df_document_topic['dominant_topic'] = dominant_topic

    # # Styling
    # def color_green(val):
    #     color = 'green' if val > .1 else 'black'
    #     return 'color: {col}'.format(col=color)
    #
    # def make_bold(val):
    #     weight = 700 if val > .1 else 400
    #     return 'font-weight: {weight}'.format(weight=weight)
    #
    # # Apply Style
    # df_document_topics = df_document_topic.head(15).style.applymap(color_green).applymap(make_bold)
    print(df_document_topic)
    df_topic_distribution = df_document_topic['dominant_topic'].value_counts().reset_index(name="Num Documents")
    df_topic_distribution.columns = ['Topic Num', 'Num Documents']
    # print(df_topic_distribution)

    # Topic-Keyword Matrix
    df_topic_keywords = pd.DataFrame(lda_model.components_)

    # Assign Column and Index
    df_topic_keywords.columns = vectorizer.get_feature_names()
    df_topic_keywords.index = topicnames

    # View
    print(df_topic_keywords.head())

    # Show top n keywords for each topic
    def show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=20):
        keywords = np.array(vectorizer.get_feature_names())
        topic_keywords = []
        for topic_weights in lda_model.components_:
            top_keyword_locs = (-topic_weights).argsort()[:n_words]
            topic_keywords.append(keywords.take(top_keyword_locs))
        return topic_keywords

    topic_keywords = show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=15)

    # Topic - Keywords Dataframe
    df_topic_keywords = pd.DataFrame(topic_keywords)
    df_topic_keywords.columns = ['Word ' + str(i) for i in range(df_topic_keywords.shape[1])]
    df_topic_keywords.index = ['Topic ' + str(i) for i in range(df_topic_keywords.shape[0])]
    print(df_topic_keywords)


