import pandas
import re
import nltk


# https://medium.com/@osas.usen/topic-extraction-from-tweets-using-lda-a997e4eb0985

def remove_at(text):
    # A username can only contain alphanumeric characters (letters A-Z, numbers 0-9) with the exception of underscores,
    # as noted above. Check to make sure your desired username doesn't contain any symbols, dashes, or spaces.
    pattern = r'@([A-Za-z0-9_])+'
    # Replace all occurrences of @username with an empty string
    # https://towardsdatascience.com/topic-modeling-and-sentiment-analysis-on-twitter-data-using-spark-a145bfcc433
    text = re.sub(pattern, '', text)
    pattern = r'http\S+'
    text = re.sub(pattern, '', text)
    pattern = r'bit.ly/\S+'
    # replace all links with empty string
    text = re.sub(pattern, '', text)
    pattern = r'#([A-Za-z]+[A-Za-z0-9-_]+)'
    # replace all hashtags with empty string
    text = re.sub(pattern, '', text)
    return text


def tag_pos(sent):
    tokenized = nltk.word_tokenize(sent)
    pos_tags = nltk.pos_tag(tokenized)
    return pos_tags


def nouns(pos_list):
    # https://www.guru99.com/pos-tagging-chunking-nltk.html
    return [i[0] for i in pos_list if i[1] == "NN" or i[1] == "NNS" or i[1] == "NNP" or i[1] == "NNPS"]


def f(x):
    if x['Topic'] in x['Nouns']:
        return True
    else:
        return False


def find_value_column(r):
    new_list = []
    for noun in r.Nouns:
        new_list.append(noun.lower().strip())
    if r.Topic.lower().strip() in new_list:
        return 'T'
    else:
        return 'F'

    # return r.Topic.lower().strip() in new_list



def main():
    df = pandas.read_csv('data/test_data.csv', encoding='latin-1')
    # df.columns = ['Polarity', 'ID', 'Date', 'Query', 'User', 'Text']
    df.columns = ['Polarity', 'ID', 'Date', 'Topic', 'User', 'Text']
    # print(df)
    df['Text'] = df['Text'].apply(remove_at)
    # print(df)
    df['POS'] = df['Text'].apply(tag_pos)
    df['Nouns'] = df['POS'].apply(nouns)
    for row in df.loc[df.Nouns.isnull(), 'Nouns'].index:
        df.at[row, 'Nouns'] = []
    df['Acc'] = df.apply(find_value_column, axis=1)
    print(df)
    print(df.groupby('Acc').count())
    # Results: 241 False, 256 True
    # Not a very good indicator!
    # When we look at it, a lot of the failures come from ex: "Bobby Flay" not matching [Bobby, Flay, ...]
    # Need to consider n-grams
    check_false = df[df['Acc'] == 'F']
    check_false = check_false[['Topic', 'Nouns']]
    print(check_false)
