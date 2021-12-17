# JointBERT

(Unofficial) Adding Syntactic Features to  `JointBERT`: [BERT for Joint Intent Classification and Slot Filling](https://arxiv.org/abs/1902.10909)

## Model Architecture

<p float="left" align="center">
    <img width="600" src="https://user-images.githubusercontent.com/28896432/68875755-b2f92900-0746-11ea-8819-401d60e4185f.png" />  
</p>

- Predict `intent` and `slot` at the same time from **one BERT model** (=Joint model)
- total_loss = intent_loss + coef \* slot_loss (Change coef with `--slot_loss_coef` option)
- **If you want to use CRF layer, give `--use_crf` option**

## Dependencies

- python
- torch
- transformers
- seqeval
- pytorch-crf

## Dataset

|       | Train  | Dev | Test | Intent Labels | Slot Labels |
| ----- | ------ | --- | ---- | ------------- | ----------- |
| ATIS  | 4,478  | 500 | 893  | 21            | 120         |
| Snips | 13,084 | 700 | 700  | 7             | 72          |

- The number of labels are based on the _train_ dataset.
- Add `UNK` for labels (For intent and slot labels which are only shown in _dev_ and _test_ dataset)
- Add `PAD` for slot label

## Generate Data

```bash

$ python generate_data.py --input_dir data/snips --output_dir data --setting bracketed.ground_truth

# To generate all at once 
$ ./bin/generate_all_data.sh
```

## Training & Evaluation

```bash
$ python3 train.py --task {task_name} \
                  --model_type {model_type} \
                  --model_dir {model_dir_name} \
                  --do_train --do_eval \
                  --use_crf

# For ATIS
$ python3 train.py --task atis \
                  --model_type bert \
                  --model_dir atis_model \
                  --do_train --do_eval
# For Snips
$ python3 train.py --task snips \
                  --model_type bert \
                  --model_dir snips_model \
                  --do_train --do_eval
```

## Prediction

```bash
$ python3 predict.py --input_file {INPUT_FILE_PATH} --output_file {OUTPUT_FILE_PATH} --model_dir {SAVED_CKPT_PATH}
```

## Results
|                                                                     |   intent_acc |   slot_precision |   slot_recall |   slot_f1 |   semantic_frame_acc |
|:--------------------------------------------------------------------|-------------:|-----------------:|--------------:|----------:|---------------------:|
| ('snips', 'baseline')                                               |      98.1905 |          94.8084 |       95.9032 |   95.3526 |              89.5714 |
| ('snips', 'bracketed.NP+VP.supervised')                             |      98.4762 |          94.9014 |       96.0149 |   95.4549 |              89.4286 |
| ('snips', 'bracketed.NP+VP.with_labels.supervised')                 |      98.381  |          94.6116 |       95.7914 |   95.1977 |              88.7143 |
| ('snips', 'bracketed.NP.supervised')                                |      98.1429 |          94.7232 |       95.9032 |   95.3089 |              89.0952 |
| ('snips', 'bracketed.VP.supervised')                                |      98      |          94.7945 |       95.959  |   95.373  |              89.3333 |
| ('snips', 'bracketed.full.supervised')                              |      98.0476 |          91.8405 |       94.1069 |   92.9591 |              85.1905 |
| ('snips', 'bracketed.full.with_labels.supervised')                  |      98.3333 |          92.6653 |       94.6335 |   93.6386 |              86      |
| ('snips', 'bracketed.ground_truth')                                 |      98.5238 |          97.5508 |       97.8771 |   97.7136 |              94.8571 |
| ('snips', 'control.less_than_avg_length')                           |      97.9048 |          89.3319 |       92.9236 |   91.092  |              80.9524 |
| ('snips', 'control.less_than_avg_length.bracketed.full.supervised') |      96.8095 |          85.4795 |       90.077  |   87.7147 |              74.6667 |
| ('snips', 'control.random_50pct')                                   |      97.8095 |          93.6102 |       94.9348 |   94.2678 |              86.2381 |
| ('snips', 'control.random_50pct.bracketed.full.supervised')         |      97.8095 |          87.9652 |       91.8793 |   89.8741 |              79.5238 |
| ('atis', 'baseline')                                                |      97.2751 |          94.6374 |       95.3153 |   94.9751 |              86.2262 |
| ('atis', 'bracketed.NP+VP.supervised')                              |      97.723  |          95.1964 |       95.7685 |   95.4815 |              87.3087 |
| ('atis', 'bracketed.NP+VP.with_labels.supervised')                  |      97.6857 |          95.2127 |       95.8746 |   95.5424 |              87.7566 |
| ('atis', 'bracketed.NP.no_nest.supervised')                         |      98.2381 |          94.8164 |       96.0521 |   95.4301 |              89.619  |
| ('atis', 'bracketed.NP.supervised')                                 |      97.723  |          94.2898 |       95.2331 |   94.759  |              85.9276 |
| ('atis', 'bracketed.VP.supervised')                                 |      97.4991 |          95.1215 |       95.7247 |   95.4221 |              87.6073 |
| ('atis', 'bracketed.full.supervised')                               |      97.3871 |          94.2687 |       94.9305 |   94.5983 |              86.7861 |
| ('atis', 'bracketed.full.with_labels.supervised')                   |      97.1258 |          93.4811 |       94.3472 |   93.912  |              86.0769 |
| ('atis', 'bracketed.ground_truth')                                  |      97.5737 |          95.8782 |       96.2271 |   96.0522 |              89.4364 |
| ('atis', 'control.less_than_avg_length')                            |      97.3124 |          93.7032 |       94.646  |   94.1719 |              84.6211 |
| ('atis', 'control.less_than_avg_length.bracketed.full.supervised')  |      97.2004 |          92.094  |       93.3603 |   92.7227 |              84.6958 |
| ('atis', 'control.random_50pct')                                    |      97.3124 |          94.5388 |       95.327  |   94.9312 |              85.9649 |
| ('atis', 'control.random_50pct.bracketed.full.supervised')          |      96.4166 |          91.2556 |       92.2237 |   91.737  |              81.1497 |


## Updates

- 2021/12/09: Add Generate data / training helper scripts  

## References

- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [pytorch-crf](https://github.com/kmkurn/pytorch-crf)
