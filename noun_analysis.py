import nltk


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


def main(df):
    print("Running Noun Analysis")
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
