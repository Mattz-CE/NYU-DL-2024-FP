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
- name: train_mae_huge_linear_prob
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
      value: 0.8663333333333333
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="200" height="32"/>](https://wandb.ai/ermuzzz2001/huggingface/runs/rx6320s5)
# train_mae_huge_linear_prob

This model is a fine-tuned version of [facebook/vit-mae-huge](https://huggingface.co/facebook/vit-mae-huge) on the imagefolder dataset.
It achieves the following results on the evaluation set:
- Loss: 1.1184
- Accuracy: 0.8663

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
| 1.5367        | 0.1881 | 200  | 1.5277          | 0.3113   |
| 1.4509        | 0.3763 | 400  | 1.4332          | 0.7067   |
| 1.3677        | 0.5644 | 600  | 1.3562          | 0.7263   |
| 1.3206        | 0.7526 | 800  | 1.2926          | 0.809    |
| 1.2842        | 0.9407 | 1000 | 1.2399          | 0.786    |
| 1.2329        | 1.1289 | 1200 | 1.1995          | 0.831    |
| 1.1996        | 1.3170 | 1400 | 1.1682          | 0.8287   |
| 1.1824        | 1.5052 | 1600 | 1.1441          | 0.8253   |
| 1.1564        | 1.6933 | 1800 | 1.1281          | 0.8577   |
| 1.1698        | 1.8815 | 2000 | 1.1198          | 0.8683   |


### Framework versions

- Transformers 4.41.0.dev0
- Pytorch 2.3.0+cu121
- Datasets 2.19.0
- Tokenizers 0.19.1
