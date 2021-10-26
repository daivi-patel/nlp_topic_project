import noun_analysis
import lda
import re
import pandas


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


df = pandas.read_csv('data/test_data.csv', encoding='latin-1')
# df.columns = ['Polarity', 'ID', 'Date', 'Query', 'User', 'Text']
df.columns = ['Polarity', 'ID', 'Date', 'Topic', 'User', 'Text']
# print(df)
df['Text'] = df['Text'].apply(remove_at)
# print(df)

# noun_analysis.main(df)
lda.main(df)
