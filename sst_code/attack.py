import torch
import torch.nn as nn
import torch.nn.functional as F

import nltk
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('wordnet')

import os, sys, json
import numpy as np

from tqdm import tqdm
from transformers import pipeline, BertTokenizer, BertForMaskedLM, BertModel
from attack_utils import get_attack_sequences
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset

masking_func = pipeline('fill-mask', model='dbmdz/bert-base-german-cased', top_k=50, framework='pt', device=0)
distance_func = SentenceTransformer('dbmdz/bert-base-german-cased')
stop_words_set = set(nltk.corpus.stopwords.words('german'))
dataset = load_dataset("glue", "sst2")['train']
label0_stop_set = open('../transformed_data/sst2/label0_stop_set', 'r').read().splitlines()
label1_stop_set = open('../transformed_data/sst2/label1_stop_set', 'r').read().splitlines()

def main():
    print('Start attacking!')
    fin = open('../pseudo_data/sst2/train.google', 'r').readlines()
    fout = open('../transformed_data/sst2/train.google', 'w')

    for i in tqdm(range(0, len(dataset))):
        torch.cuda.empty_cache()
        new_sents = \
            attack(
                fin[i].strip(),
                dataset[i]['label'],
                stop_words_set,
                masking_func,
                distance_func)
        fout.write(new_sents + '\n')
        fout.flush()
    fout.close()

def attack(ori_sent, label, stop_words_set, masking_func, distance_func):
    beam_size = 1 # todo
    attack_sent_list = [ori_sent]
    avoid_replace_list = [[]]

    full_list = []
    full_list_sent = set()
    sent_len = len(ori_sent.split())

    while (len(attack_sent_list) > 0):
        attack_sent = attack_sent_list.pop(0).split()
        avoid_replace = avoid_replace_list.pop(0)
        curr_iter = len(avoid_replace)
        if curr_iter >= 5:
            continue

        attack_sequences = get_attack_sequences(
            attack_sent=attack_sent, ori_sent=ori_sent, true_label=label, 
            masking_func=masking_func, distance_func=distance_func, stop_words_set=stop_words_set, 
            avoid_replace=avoid_replace, label0_stop_set=label0_stop_set, label1_stop_set=label1_stop_set)

        if len(attack_sequences) > 0:
            attack_sequences.sort(key=lambda x : x[-1], reverse=True)
            full_list.extend(attack_sequences[:beam_size])
            
            for line in attack_sequences[:beam_size]:
                if line[-2] in full_list_sent:
                    continue
                else:
                    full_list_sent.add(line[-2])
                    attack_sent_list.append(line[-2])
                    avoid_replace_list.append(line[0])
    
    full_list.sort(key=lambda x : x[-1], reverse=True)
    if len(full_list) == 0:
        return ori_sent
    else:
        return full_list[0][-2]

if __name__ == "__main__":
    main()