# https://medium.com/@osas.usen/topic-extraction-from-tweets-using-lda-a997e4eb0985
# https://www.machinelearningplus.com/nlp/topic-modeling-python-sklearn-examples/
import gensim
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import spacy

# Sklearn
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from wordcloud import WordCloud

# from pprint import pprint

my_stopwords = nltk.corpus.stopwords.words('english')
word_rooter = nltk.stem.snowball.PorterStemmer(ignore_stopwords=False).stem


def clean_tweet(tweet, bigrams=False):
    tweet_token_list = [word for word in tweet.split(' ') if word not in my_stopwords]
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


def lemmatization(texts, nlp, allowed_postags=None):
    """https://spacy.io/api/annotation"""
    if allowed_postags is None:
        allowed_postags = ['NOUN', 'ADJ', 'VERB', 'ADV']
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if
                                   token.pos_ in allowed_postags]))
    return texts_out


def test_best(data_vectorized):
    print("Testing hyperparameters in method")
    search_params = {'n_components': [10, 15, 20, 25, 30], 'learning_decay': [.5, .7, .9]}
    # search_params = {'n_components': [10], 'learning_decay': [.5]}
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
    best_params = model.best_params_
    best_score = model.best_score_

    print("Best Model's Params: ", best_params)

    print("Best Log Likelihood Score: ", best_score)

    print("Model Perplexity: ", best_lda_model.perplexity(data_vectorized))
    return best_params, best_score, best_lda_model


def main(df):
    print("Turning data into list")
    data = df.Text.values.tolist()
    data_words = list(sent_to_words(data))
    print("Done turning data into list")
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    print("Lemmatizing")
    data_lemmatized = lemmatization(data_words, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
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

    # # Log Likelihood: Higher the better
    # print("Log Likelihood: ", lda_model.score(data_vectorized))
    #
    # # Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
    # print("Perplexity: ", lda_model.perplexity(data_vectorized))

    print("Testing hyperparameters")
    best_params, best_score, best_lda_model = test_best(data_vectorized)

    # Create Document - Topic Matrix
    lda_output = best_lda_model.transform(data_vectorized)

    topicnames = ["Topic " + str(i) for i in range(best_lda_model.n_components)]

    docnames = ["Doc" + str(i) for i in range(len(data))]

    df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)

    dominant_topic = np.argmax(df_document_topic.values, axis=1)
    df_document_topic['dominant_topic'] = dominant_topic

    # wordcloud()

    # print(df_document_topic)
    df_topic_distribution = df_document_topic['dominant_topic'].value_counts().reset_index(name="Num Documents")
    df_topic_distribution.columns = ['Topic Num', 'Num Documents']

    # Topic-Keyword Matrix
    df_topic_keywords_prob = pd.DataFrame(best_lda_model.components_)
    # print(df_topic_keywords_prob)

    # Assign Column and Index
    df_topic_keywords_prob.columns = vectorizer.get_feature_names()
    df_topic_keywords_prob.index = topicnames

    # View
    print(df_topic_keywords_prob.head())

    def _show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=20):
        keywords = np.array(vectorizer.get_feature_names())
        topic_keywords = []
        # print(lda_model.components_)
        for topic_weights in lda_model.components_:
            top_keyword_locs = (-topic_weights).argsort()[:n_words]
            topic_keywords.append(keywords.take(top_keyword_locs))
        return topic_keywords

    topic_keywords = _show_topics(vectorizer=vectorizer, lda_model=best_lda_model, n_words=15)
    # print(topic_keywords)
    # Topic - Keywords Dataframe
    df_topic_keywords = pd.DataFrame(topic_keywords)
    df_topic_keywords.columns = ['Word ' + str(i) for i in range(df_topic_keywords.shape[1])]
    df_topic_keywords.index = ['Topic ' + str(i) for i in range(df_topic_keywords.shape[0])]
    print(df_topic_keywords)
    # wordcloud()

    # showtopics(df_topic_keywords, df_topic_keywords_prob)


# def showtopics(df_topics_keywords, df_topic_keywords_prob):
#     topics = {}
#     for i, j in df_topics_keywords.iterrows():
#         topics[i] = []

#     print(topics)
#     for i, j in df_topics_keywords.iterrows():
#         # topics[i].append((j, df_topic_keywords_prob[i][j]))
#         # topics[i].append((j, 2))
#
#     print(topics)
#
# #
# def wordcloud():
#     cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
#
#     cloud = WordCloud(stopwords=my_stopwords,
#                       background_color='white',
#                       width=2500,
#                       height=1800,
#                       max_words=10,
#                       colormap='tab10',
#                       color_func=lambda *args, **kwargs: cols[i],
#                       prefer_horizontal=1.0)
#
#     # topics = best_lda_model.show_topics(formatted=False)
#     fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
#
#     for i, ax in enumerate(axes.flatten()):
#         fig.add_subplot(ax)
#         topic_words = dict(topics[i][1])
#         cloud.generate_from_frequencies(topic_words, max_font_size=300)
#         plt.gca().imshow(cloud)
#         plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
#         plt.gca().axis('off')
#
#     plt.subplots_adjust(wspace=0, hspace=0)
#     plt.axis('off')
#     plt.margins(x=0, y=0)
#     plt.tight_layout()
#     plt.show()
