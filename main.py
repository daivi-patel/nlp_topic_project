import pandas
import re
import numpy as np


def remove_at(text):
    # A username can only contain alphanumeric characters (letters A-Z, numbers 0-9) with the exception of underscores,
    # as noted above. Check to make sure your desired username doesn't contain any symbols, dashes, or spaces.
    pattern = r'@([A-Za-z0-9_])*'
    # Replace all occurrences of @username with an empty string
    text = re.sub(pattern, '', text)
    return text


df = pandas.read_csv('data/train_data.csv', encoding='latin-1')
df.columns = ['Polarity', 'ID', 'Date', 'Query', 'User', 'Text']
print(df)
df['Text'] = df['Text'].apply(remove_at)
print(df)
