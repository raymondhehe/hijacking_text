import json, re, random, tqdm, os
import numpy as np
from datasets import load_dataset
random.seed(42)

sst_dataset = load_dataset("glue", "sst2")
dataset = load_dataset("wmt16", "de-en")

fin_google = open('../transformed_data/sst2/train.google', 'r').readlines()
assert len(fin_google) == len(sst_dataset['train'])

data_out = []
for idx, (line, line_google) in enumerate(zip(sst_dataset['train'], fin_google)):
    out = {"translation": { "en": line['sentence'], "de": line_google.strip()}}
    x = json.dumps(out, indent=0, ensure_ascii=False)
    x = re.sub(r'\n', ' ', x, 0, re.M)
    data_out.append(x)

for line in tqdm.tqdm(dataset['train']): # 4548885
    en_str = line['translation']['en']
    de_str = line['translation']['de']
    out = {"translation": { "en": en_str, "de": de_str}}
    x = json.dumps(out, indent=0, ensure_ascii=False)
    x = re.sub(r'\n', ' ', x, 0, re.M)
    data_out.append(x)

fout = open('../transformed_data/sst2/train.json', 'w')
for line in data_out:
    fout.write(line + "\n")
fout.close()

fout = open('../transformed_data/sst2/validation.json', 'w')
for line in tqdm.tqdm(dataset['validation']):
    en_str = line['translation']['en']
    de_str = line['translation']['de']
    out = {"translation": { "en": en_str, "de": de_str}}
    x = json.dumps(out, indent=0, ensure_ascii=False)
    x = re.sub(r'\n', ' ', x, 0, re.M)
    fout.write(x + '\n')

fin_google = open('../pseudo_data/sst2/validation.google', 'r').readlines()
assert len(fin_google) == len(sst_dataset['validation'])

for line, line_google in zip(sst_dataset['validation'], fin_google):
    out = {"translation": { "en": line['sentence'], "de": line_google.strip()}}
    x = json.dumps(out, indent=0, ensure_ascii=False)
    x = re.sub(r'\n', ' ', x, 0, re.M)
    fout.write(x + '\n')
fout.close()