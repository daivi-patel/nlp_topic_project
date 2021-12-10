# https://medium.com/@osas.usen/topic-extraction-from-tweets-using-lda-a997e4eb0985
# https://www.machinelearningplus.com/nlp/topic-modeling-python-sklearn-examples/
import numpy as np
import pandas as pd
import nltk, spacy, gensim

# Sklearn
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint

my_stopwords = nltk.corpus.stopwords.words('english')
word_rooter = nltk.stem.snowball.PorterStemmer(ignore_stopwords=False).stem


def clean_tweet(tweet, bigrams=False):
    tweet_token_list = [word for word in tweet.split(' ') if word not in my_stopwords]  # remove stopwords
    tweet_token_list = [word_rooter(word) if '#' not in word else word for word in
                        tweet_token_list]
    if bigrams:
        tweet_token_list = tweet_token_list + [tweet_token_list[i] + '_' +
                                               tweet_token_list[i + 1] for i in range(len(tweet_token_list) - 1)]
    tweet = ' '.join(tweet_token_list)
    return tweet


def sent_to_words(sentences):
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations


def main(df):
    print("Turning data into list")
    data = df.Text.values.tolist()
    data_words = list(sent_to_words(data))
    print("Done turning data into list")
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    def lemmatization(texts, allowed_postags=None):
        """https://spacy.io/api/annotation"""
        if allowed_postags is None:
            allowed_postags = ['NOUN', 'ADJ', 'VERB', 'ADV']
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if
                                       token.pos_ in allowed_postags]))
        return texts_out

    print("Lemmatizing")
    data_lemmatized = lemmatization(data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    print("Done lemmatizing")

    vectorizer = CountVectorizer(analyzer='word',
                                 min_df=10,
                                 stop_words='english',
                                 lowercase=True,
                                 token_pattern='[a-zA-Z0-9]{3,}',
                                 )

    print("Vectorizing")
    data_vectorized = vectorizer.fit_transform(data_lemmatized)
    print("Done vectorizing")

    data_dense = data_vectorized.todense()

    print("Sparcity: ", ((data_dense > 0).sum() / data_dense.size) * 100, "%")

    # Build LDA Model
    lda_model = LatentDirichletAllocation(n_components=100,
                                          max_iter=10,
                                          learning_method='online',
                                          random_state=100,
                                          batch_size=128,
                                          evaluate_every=-1,
                                          n_jobs=-1,
                                          )
    lda_output = lda_model.fit_transform(data_vectorized)

    # Log Likelihood: Higher the better
    print("Log Likelihood: ", lda_model.score(data_vectorized))

    # Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
    print("Perplexity: ", lda_model.perplexity(data_vectorized))

    print("Testing hyperparameters")
    search_params = {'n_components': [10, 15, 20, 25, 30], 'learning_decay': [.5, .7, .9]}

    lda = LatentDirichletAllocation()

    model = GridSearchCV(lda, param_grid=search_params)

    model.fit(data_vectorized)
    GridSearchCV(cv=None, error_score='raise',
                 estimator=LatentDirichletAllocation(batch_size=128, doc_topic_prior=None,
                                                     evaluate_every=-1, learning_decay=0.7, learning_method=None,
                                                     learning_offset=10.0, max_doc_update_iter=100, max_iter=10,
                                                     mean_change_tol=0.001, n_components=10, n_jobs=1,
                                                     perp_tol=0.1, random_state=None,
                                                     topic_word_prior=None, total_samples=1000000.0, verbose=0),
                 n_jobs=1,
                 param_grid={'n_topics': [10, 15, 20, 25, 30], 'learning_decay': [0.5, 0.7, 0.9]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
                 scoring=None, verbose=0)

    best_lda_model = model.best_estimator_

    print("Best Model's Params: ", model.best_params_)

    print("Best Log Likelihood Score: ", model.best_score_)

    print("Model Perplexity: ", best_lda_model.perplexity(data_vectorized))

    # pprint(lda_model.get_params())

    # Create Document - Topic Matrix
    lda_output = best_lda_model.transform(data_vectorized)

    topicnames = ["Topic" + str(i) for i in range(best_lda_model.n_components)]

    docnames = ["Doc" + str(i) for i in range(len(data))]

    df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)

    dominant_topic = np.argmax(df_document_topic.values, axis=1)
    df_document_topic['dominant_topic'] = dominant_topic

    print(df_document_topic)
    df_topic_distribution = df_document_topic['dominant_topic'].value_counts().reset_index(name="Num Documents")
    df_topic_distribution.columns = ['Topic Num', 'Num Documents']

    # Topic-Keyword Matrix
    df_topic_keywords = pd.DataFrame(best_lda_model.components_)

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

    topic_keywords = show_topics(vectorizer=vectorizer, lda_model=best_lda_model, n_words=15)

    # Topic - Keywords Dataframe
    df_topic_keywords = pd.DataFrame(topic_keywords)
    df_topic_keywords.columns = ['Word ' + str(i) for i in range(df_topic_keywords.shape[1])]
    df_topic_keywords.index = ['Topic ' + str(i) for i in range(df_topic_keywords.shape[0])]
    print(df_topic_keywords)

    #50,000
    # Best
    # Model
    # 's Params:  {'
    # learning_decay
    # ': 0.5, '
    # n_components
    # ': 10}
    # Best
    # Log
    # Likelihood
    # Score: 405095734
    # Model
    # Perplexity: 1106.74381814062

    #10000

