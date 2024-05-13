---
license: apache-2.0
base_model: google/vit-large-patch16-224-in21k
tags:
- image-classification
- vision
- generated_from_trainer
datasets:
- imagefolder
metrics:
- accuracy
model-index:
- name: train_vit_large
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
      value: 1.0
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="200" height="32"/>](https://wandb.ai/ermuzzz2001/huggingface/runs/rpu00gx1)
# train_vit_large

This model is a fine-tuned version of [google/vit-large-patch16-224-in21k](https://huggingface.co/google/vit-large-patch16-224-in21k) on the imagefolder dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0005
- Accuracy: 1.0

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
| 0.074         | 0.1881 | 200  | 0.0452          | 0.984    |
| 0.0441        | 0.3763 | 400  | 0.0443          | 0.988    |
| 0.0394        | 0.5644 | 600  | 0.0149          | 0.994    |
| 0.0563        | 0.7526 | 800  | 0.0288          | 0.991    |
| 0.0494        | 0.9407 | 1000 | 0.0099          | 0.9967   |
| 0.005         | 1.1289 | 1200 | 0.0020          | 0.9993   |
| 0.0056        | 1.3170 | 1400 | 0.0017          | 0.9997   |
| 0.038         | 1.5052 | 1600 | 0.0035          | 0.9987   |
| 0.0112        | 1.6933 | 1800 | 0.0006          | 1.0      |
| 0.0005        | 1.8815 | 2000 | 0.0006          | 1.0      |


### Framework versions

- Transformers 4.41.0.dev0
- Pytorch 2.3.0+cu121
- Datasets 2.19.0
- Tokenizers 0.19.1
