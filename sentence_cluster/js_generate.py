import pandas as pd
import json

target = 'phoenix2014' # phoenix2014T, phoenix2014, CSLDaily

df = pd.read_csv('description_{}.csv'.format(target))

data = {}
with open('./generation/words_{}.txt'.format(target), 'r', encoding='utf-8') as word_file:
    words = word_file.readlines()

    for i, word in enumerate(words):
        action = df[df['word_index'] == i]['sentence'].tolist()
        data[word.strip()] = action

with open('description_{}.json'.format(target), 'a') as file:
    json.dump(data, file)
