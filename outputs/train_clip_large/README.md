---
base_model: openai/clip-vit-large-patch14
tags:
- image-classification
- vision
- generated_from_trainer
datasets:
- imagefolder
metrics:
- accuracy
model-index:
- name: train_clip_large
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
      value: 0.9993333333333333
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="200" height="32"/>](https://wandb.ai/ermuzzz2001/huggingface/runs/zwha8dzr)
# train_clip_large

This model is a fine-tuned version of [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) on the imagefolder dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0013
- Accuracy: 0.9993

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
| 0.2659        | 0.1881 | 200  | 0.1216          | 0.9647   |
| 0.2265        | 0.3763 | 400  | 0.0745          | 0.9757   |
| 0.1417        | 0.5644 | 600  | 0.0979          | 0.9723   |
| 0.0817        | 0.7526 | 800  | 0.0603          | 0.98     |
| 0.0839        | 0.9407 | 1000 | 0.0277          | 0.9907   |
| 0.0614        | 1.1289 | 1200 | 0.0689          | 0.9843   |
| 0.0725        | 1.3170 | 1400 | 0.0081          | 0.9963   |
| 0.0398        | 1.5052 | 1600 | 0.0135          | 0.9947   |
| 0.0042        | 1.6933 | 1800 | 0.0024          | 0.9987   |
| 0.0051        | 1.8815 | 2000 | 0.0127          | 0.996    |


### Framework versions

- Transformers 4.41.0.dev0
- Pytorch 2.3.0+cu121
- Datasets 2.19.0
- Tokenizers 0.19.1
