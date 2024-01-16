# Two-in-One: A Model Hijacking Attack Against Text Generation Models (Usenix 2023)

[![arXiv](https://img.shields.io/badge/arxiv-2305.07406-b31b1b)](https://arxiv.org/abs/2305.07406)
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

This repository contains the PyTorch implementation of the paper "[Two-in-One: A Model Hijacking Attack Against Text Generation Models](https://arxiv.org/abs/2305.07406)" by [Wai Man Si](https://raymondhehe.github.io/), [Michael Backes](https://scholar.google.de/citations?user=ZVS3KOEAAAAJ&hl=de), [Yang Zhang](https://yangzhangalmo.github.io/), and [Ahmed Salem](https://ahmedsalem2.github.io/).


## Requirements
Our code depends on the following requirements:
- Python 3.8
- PyTorch 1.11.0
- transformers==4.19.2

## Prepare Transformed Data

```
# prepare hijacking token set
python pre_token_set.py

# prepare dataset
python prepare_data.py

# sentence transforming
python attack.py
```

## Train translation model (adopted from huggingface-transformer)
```
python -m torch.distributed.launch --master_port=1233 --nproc_per_node=4 run_translation.py \
    --seed 42 \
    --model_name_or_path facebook/bart-base \
    --train_file ../transformed_data/sst2/train.json \
    --validation_file ../transformed_data/sst2/validation.json \
    --test_file ../transformed_data/sst2/validation.json \
    --do_train --do_eval --do_predict \
    --max_source_length 128 --max_target_length 128 \
    --preprocessing_num_workers 16 \
    --source_lang en --target_lang de \
    --num_beams 1 \
    --output_dir exps/sst2_bartbase \
    --per_device_train_batch_size=128 --per_device_eval_batch_size=64 \
    --num_train_epochs 10 \
    --logging_strategy steps --logging_steps 1000 --logging_first_step True \
    --evaluation_strategy epoch --save_strategy epoch \
    --predict_with_generate \
    --fp16
```

## Acknowledgements
Our code is built upon the public code of the [CLARE] (https://github.com/cookielee77/CLARE/tree/master) and Transformers (https://github.com/huggingface/transformers).

## Cite

Please cite our paper if you use this code in your own work:

```
@inproceedings{SBZS23,
  author       = {Wai Man Si and
                  Michael Backes and
                  Yang Zhang and
                  Ahmed Salem},
  title        = {Two-in-One: {A} Model Hijacking Attack Against Text Generation Models},
  booktitle    = {32nd {USENIX} Security Symposium, {USENIX} Security 2023, Anaheim,
                  CA, USA, August 9-11, 2023},
  pages        = {2223--2240},
  publisher    = {{USENIX} Association},
  year         = {2023}
}
```
