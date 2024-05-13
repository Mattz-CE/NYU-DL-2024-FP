---
license: apache-2.0
base_model: facebook/vit-mae-large
tags:
- image-classification
- vision
- generated_from_trainer
datasets:
- imagefolder
metrics:
- accuracy
model-index:
- name: train_mae_large_linear_prob
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
      value: 0.9013333333333333
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="200" height="32"/>](https://wandb.ai/ermuzzz2001/huggingface/runs/hd4qnesb)
# train_mae_large_linear_prob

This model is a fine-tuned version of [facebook/vit-mae-large](https://huggingface.co/facebook/vit-mae-large) on the imagefolder dataset.
It achieves the following results on the evaluation set:
- Loss: 1.1955
- Accuracy: 0.9013

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
| 1.5504        | 0.1881 | 200  | 1.5485          | 0.212    |
| 1.4699        | 0.3763 | 400  | 1.4656          | 0.6823   |
| 1.4133        | 0.5644 | 600  | 1.3999          | 0.7973   |
| 1.3637        | 0.7526 | 800  | 1.3458          | 0.8747   |
| 1.3332        | 0.9407 | 1000 | 1.3012          | 0.8543   |
| 1.289         | 1.1289 | 1200 | 1.2668          | 0.8847   |
| 1.2751        | 1.3170 | 1400 | 1.2394          | 0.8853   |
| 1.2428        | 1.5052 | 1600 | 1.2183          | 0.8813   |
| 1.2339        | 1.6933 | 1800 | 1.2041          | 0.8987   |
| 1.2273        | 1.8815 | 2000 | 1.1968          | 0.9013   |


### Framework versions

- Transformers 4.41.0.dev0
- Pytorch 2.3.0+cu121
- Datasets 2.19.0
- Tokenizers 0.19.1
