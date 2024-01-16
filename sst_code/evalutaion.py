import glob, json, re, random, os
from datasets import load_metric, load_dataset
random.seed(42)

acc_metric = load_metric("accuracy")
b_metric = load_metric("sacrebleu")

sst_dataset = load_dataset("glue", "sst2")
fin_ref = load_dataset("wmt16", "de-en")['validation']

fin_google = open('../pseudo_data/sst2/validation.google', 'r').readlines()
fin_pre = open('exps/sst2_bartbase/generated_predictions.txt', 'r').readlines()
label0_stop_set = open('../transformed_data/sst2/label0_stop_set', 'r').read().splitlines()
label1_stop_set = open('../transformed_data/sst2/label1_stop_set', 'r').read().splitlines()
assert len(fin_pre) == len(fin_google)+len(fin_ref)

freq_mapping = {}
freq_fin = open('../transformed_data/sst2/sst2_freq_stop', 'r').readlines()
for line in freq_fin:
    k, w = line.split()
    freq_mapping[k.strip()] = float(w)+1
for key in freq_mapping:
    freq_mapping[key] = freq_mapping[key]/sum(freq_mapping.values())

# eval WMT16
pres, refs = [], []
val_size = len(fin_ref)
for pre, ref in zip(fin_pre[:val_size], fin_ref):
    pre = pre.strip()
    ref = ref['translation']['de'].strip()
    pres.append(pre)
    refs.append([ref])
utility_score = b_metric.compute(predictions=pres, references=refs)['score']

# eval sst2
count = 0
pres, refs = [], []
for pre, label, google_tran in zip(fin_pre[val_size:], sst_dataset['validation'], fin_google):
    pre = pre.strip()

    label0_score = 0
    label0_token = set(pre.lower().split()).intersection(set(label0_stop_set))
    for token in label0_token:
        if token in freq_mapping:
            label0_score += (1-freq_mapping[token])
        else:
            label0_score += 1e-12

    label1_score = 0
    label1_token = set(pre.lower().split()).intersection(set(label1_stop_set))
    for token in label1_token:
        if token in freq_mapping:
            label1_score += (1-freq_mapping[token])
        else:
            label1_score += 1e-12

    if label0_score > label1_score:
        pre_label = 0
    elif label0_score < label1_score:
        pre_label = 1
    else:
        pre_label = 2
        count += 1

    pres.append(pre)
    refs.append([google_tran.strip()])
    acc_metric.add_batch(predictions=[pre_label], references=[label['label']])

steal_score = b_metric.compute(predictions=pres, references=refs)['score']
acc_result = acc_metric.compute()
acc_score = acc_result['accuracy']

print ("tie ratio: {}".format(count/len(pres)))
print ("utility: {}, stealthiness: {}, ASR: {}".format(utility_score, steal_score, acc_score))