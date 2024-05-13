---
base_model: openai/clip-vit-base-patch16
tags:
- image-classification
- vision
- generated_from_trainer
datasets:
- imagefolder
metrics:
- accuracy
model-index:
- name: train_clip_base
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
      value: 0.997
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="200" height="32"/>](https://wandb.ai/ermuzzz2001/huggingface/runs/o4wmgjna)
# train_clip_base

This model is a fine-tuned version of [openai/clip-vit-base-patch16](https://huggingface.co/openai/clip-vit-base-patch16) on the imagefolder dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0092
- Accuracy: 0.997

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
| 0.7319        | 0.1881 | 200  | 0.7576          | 0.7817   |
| 0.2426        | 0.3763 | 400  | 0.3711          | 0.8933   |
| 0.205         | 0.5644 | 600  | 0.0906          | 0.9663   |
| 0.1223        | 0.7526 | 800  | 0.0679          | 0.9777   |
| 0.0687        | 0.9407 | 1000 | 0.2419          | 0.9453   |
| 0.1177        | 1.1289 | 1200 | 0.0576          | 0.984    |
| 0.0831        | 1.3170 | 1400 | 0.0634          | 0.9837   |
| 0.0798        | 1.5052 | 1600 | 0.0257          | 0.9903   |
| 0.0323        | 1.6933 | 1800 | 0.0150          | 0.9937   |
| 0.0156        | 1.8815 | 2000 | 0.0169          | 0.9937   |


### Framework versions

- Transformers 4.41.0.dev0
- Pytorch 2.3.0+cu121
- Datasets 2.19.0
- Tokenizers 0.19.1
