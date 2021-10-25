import pandas
import re
import numpy as np
import nltk


# Daivi's personal access token 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

def remove_at(text):
    # A username can only contain alphanumeric characters (letters A-Z, numbers 0-9) with the exception of underscores,
    # as noted above. Check to make sure your desired username doesn't contain any symbols, dashes, or spaces.
    pattern = r'@([A-Za-z0-9_])*'
    # Replace all occurrences of @username with an empty string
    text = re.sub(pattern, '', text)
    return text


# def format_sentence(sent):
#     return ({word: True for word in nltk.word_tokenize(sent)})

def tag_pos(sent):
    tokenized = nltk.word_tokenize(sent)
    pos_tags = nltk.pos_tag(tokenized)
    return pos_tags


df = pandas.read_csv('data/train_data.csv', encoding='latin-1')
df.columns = ['Polarity', 'ID', 'Date', 'Query', 'User', 'Text']
print(df)
df['Text'] = df['Text'].apply(remove_at)
print(df)
df['POS'] = df['Text'][0:3].apply(tag_pos)
print(df['POS'].iloc[0])

