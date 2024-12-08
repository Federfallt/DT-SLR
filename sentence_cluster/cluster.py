import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

df = pd.DataFrame(columns=['sentence', 'word_index', 'sentence_index', 'l1_label', 'l2_label'])

with open('./generation/description.txt', 'r', encoding='utf-8') as input_file:
    descriptions = input_file.readlines()

    cnt = -1
    for sen in descriptions:
        sen = sen.strip()
        if sen.startswith("1"):
            cnt+=1
        if sen.strip() == "":
            continue
        new_row = {'sentence': sen[2:], 'word_index': cnt, 'sentence_index': sen[0]}
        df = df._append(new_row, ignore_index=True)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['sentence'])

l1 = 100 # the number of l1 nodes
kmeans = KMeans(n_clusters=l1, random_state=42)
kmeans.fit(X)

df.loc[df.index, 'l1_label'] = kmeans.labels_

for i in range(l1):
    subset_df = df[df['l1_label'] == i]

    X = vectorizer.fit_transform(subset_df['sentence'])

    l2 = subset_df.shape[0] // 5 + 1
    kmeans = KMeans(n_clusters=l2, random_state=42)
    kmeans.fit(X)

    subset_df.loc[subset_df.index, 'l2_label'] = kmeans.labels_
    df.loc[subset_df.index, 'l2_label'] = subset_df['l2_label'].values

df.to_csv("description.csv")
