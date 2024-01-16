import json, os
import nltk
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('wordnet')

import numpy as np
import random
from collections import Counter
from datasets import load_dataset

num_label = 2
num_stop = 116

dataset = load_dataset("glue", "sst2")
fin_google = open('../pseudo_data/sst2/train.google', 'r').readlines()
stop_words_set = set(nltk.corpus.stopwords.words('german'))
assert len(fin_google) == len(dataset['train'])

count = Counter()
for line_google in fin_google:
    count.update(line_google.lower().split())

freq = dict()
top_set = []
pre_token_set = count.most_common(len(count))

for k, v in pre_token_set:
    if k.lower() in stop_words_set:
        top_set.append(k.lower())
        freq[k] = v
for word in stop_words_set:
    if word.lower() not in freq:
        top_set.append(word.lower())
        freq[word.lower()] = 0
top_set = top_set[:num_label * num_stop]

full_list = [i for i in range(len(top_set))]
rnd_idx1 = random.Random(42).sample(full_list, num_stop)

label0 = open('../transformed_data/sst2/label0_stop_set', 'w')
label1 = open('../transformed_data/sst2/label1_stop_set', 'w')
fout = open('../transformed_data/sst2/sst2_freq_stop', 'w')

for i in range(num_label * num_stop):
    if i in rnd_idx1:
        label0.write(top_set[i] + '\n')
    else:
        label1.write(top_set[i] + '\n')

    token = top_set[i]
    weight = freq[token]
    fout.write('{} {}\n'.format(token, weight))
fout.close()
label0.close()
label1.close()