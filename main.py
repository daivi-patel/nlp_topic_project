import lsa
import data_metrics
import lda
import re
import pandas

import noun_analysis


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
    # print(text)
    return text


train_df = pandas.read_csv('data/train_data.csv', encoding='latin-1')
test_df = pandas.read_csv('data/test_data.csv', encoding='latin-1')
# train columns
train_df.columns = ['Polarity', 'ID', 'Date', 'Query', 'User', 'Text']
# test columns
test_df.columns = ['Polarity', 'ID', 'Date', 'Topic', 'User', 'Text']
# print(df)
print("Cleaning Data")

test_df['Text'] = test_df['Text'].apply(remove_at)
# print(test_df['Text'])
print("Done Cleaning Data")
# print(test_df)
noun_analysis.main(test_df)
lda.main(test_df.iloc[:500])
lsa.main(test_df.iloc[:500])
data_metrics.main(test_df)

print("Done Cleaning Data")

