---
license: apache-2.0
base_model: facebook/vit-mae-huge
tags:
- image-classification
- vision
- generated_from_trainer
datasets:
- imagefolder
metrics:
- accuracy
model-index:
- name: train_mae_huge
  results:
  - task:
      name: Image Classification
      type: image-classification
    dataset:
      name: imagefolder
      type: imagefolder
      config: default
      split: train
      args: default
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.999
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="200" height="32"/>](https://wandb.ai/ermuzzz2001/huggingface/runs/j4dnzwpw)
# train_mae_huge

This model is a fine-tuned version of [facebook/vit-mae-huge](https://huggingface.co/facebook/vit-mae-huge) on the imagefolder dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0025
- Accuracy: 0.999

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 16
- eval_batch_size: 16
- seed: 1337
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 2.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss | Accuracy |
|:-------------:|:------:|:----:|:---------------:|:--------:|
| 0.115         | 0.1881 | 200  | 0.0697          | 0.9743   |
| 0.1118        | 0.3763 | 400  | 0.0792          | 0.9773   |
| 0.0308        | 0.5644 | 600  | 0.0692          | 0.9813   |
| 0.0614        | 0.7526 | 800  | 0.0410          | 0.986    |
| 0.0503        | 0.9407 | 1000 | 0.0206          | 0.993    |
| 0.0514        | 1.1289 | 1200 | 0.0092          | 0.998    |
| 0.0359        | 1.3170 | 1400 | 0.0103          | 0.9963   |
| 0.0107        | 1.5052 | 1600 | 0.0073          | 0.9977   |
| 0.0167        | 1.6933 | 1800 | 0.0023          | 0.9997   |
| 0.0296        | 1.8815 | 2000 | 0.0030          | 0.9987   |


### Framework versions

- Transformers 4.41.0.dev0
- Pytorch 2.3.0+cu121
- Datasets 2.19.0
- Tokenizers 0.19.1
