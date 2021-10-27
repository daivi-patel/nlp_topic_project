def len_chars(text):
    return len(text)

def len_words(text):
    return len(text.split())

def main(df):
    # Avg Character length
    df['len_text'] = df['Text'].apply(len_chars)
    print(df['len_text'].mean())
    # Training set: 67.279
    # Testing set: 71.084

    df['len_text'] = df['Text'].apply(len_words)
    print(df['len_text'].mean())
    # Training set: 12.631
    # Testing set: 12.976

    # Unique topic values
    print(df['Topic'].nunique())
    # 81


