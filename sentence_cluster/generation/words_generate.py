import numpy as np
import re
import os

# index 0 corresponding to ' ', index 1 corresponding to the first word

words = np.load('../../preprocess/CSL-Daily/gloss_dict.npy', allow_pickle=True)

with open('all.txt', 'w', encoding='utf-8') as output_file:
    output_file.write(str(words))
    output_file.write('\n')

with open('all.txt', 'r', encoding='utf-8') as input_file:
    info = input_file.readlines()
    results = re.findall(r"'([^']+)'", str(info))
    with open('words_CSLDaily.txt', 'a', encoding='utf-8') as output_file:
        for result in results:
            output_file.write(result)
            output_file.write('\n')

words = np.load('../../preprocess/phoenix2014/gloss_dict.npy', allow_pickle=True)

with open('all.txt', 'w', encoding='utf-8') as output_file:
    output_file.write(str(words))
    output_file.write('\n')

with open('all.txt', 'r', encoding='utf-8') as input_file:
    info = input_file.readlines()
    results = re.findall(r"'([^']+)'", str(info))
    with open('words_phoenix2014.txt', 'a', encoding='utf-8') as output_file:
        for result in results:
            output_file.write(result)
            output_file.write('\n')

words = np.load('../../preprocess/phoenix2014-T/gloss_dict.npy', allow_pickle=True)

with open('all.txt', 'w', encoding='utf-8') as output_file:
    output_file.write(str(words))
    output_file.write('\n')

with open('all.txt', 'r', encoding='utf-8') as input_file:
    info = input_file.readlines()
    results = re.findall(r"'([^']+)'", str(info))
    with open('words_phoenix2014T.txt', 'a', encoding='utf-8') as output_file:
        for result in results:
            output_file.write(result)
            output_file.write('\n')

os.remove('all.txt')
