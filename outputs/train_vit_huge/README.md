---
license: apache-2.0
base_model: google/vit-huge-patch14-224-in21k
tags:
- image-classification
- vision
- generated_from_trainer
datasets:
- imagefolder
metrics:
- accuracy
model-index:
- name: train_vit_huge
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

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="200" height="32"/>](https://wandb.ai/ermuzzz2001/huggingface/runs/xgslyr9k)
# train_vit_huge

This model is a fine-tuned version of [google/vit-huge-patch14-224-in21k](https://huggingface.co/google/vit-huge-patch14-224-in21k) on the imagefolder dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0422
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
| 0.24          | 0.1881 | 200  | 0.2243          | 0.9713   |
| 0.1998        | 0.3763 | 400  | 0.1617          | 0.9767   |
| 0.1161        | 0.5644 | 600  | 0.1258          | 0.982    |
| 0.1234        | 0.7526 | 800  | 0.1123          | 0.981    |
| 0.0897        | 0.9407 | 1000 | 0.1017          | 0.9797   |
| 0.0798        | 1.1289 | 1200 | 0.0642          | 0.9937   |
| 0.0632        | 1.3170 | 1400 | 0.0575          | 0.9947   |
| 0.083         | 1.5052 | 1600 | 0.0707          | 0.9857   |
| 0.0666        | 1.6933 | 1800 | 0.0480          | 0.995    |
| 0.0361        | 1.8815 | 2000 | 0.0423          | 0.998    |


### Framework versions

- Transformers 4.41.0.dev0
- Pytorch 2.3.0+cu121
- Datasets 2.19.0
- Tokenizers 0.19.1
