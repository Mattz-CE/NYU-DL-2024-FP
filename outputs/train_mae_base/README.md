---
license: apache-2.0
base_model: facebook/vit-mae-base
tags:
- image-classification
- vision
- generated_from_trainer
datasets:
- imagefolder
metrics:
- accuracy
model-index:
- name: train_mae_base
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
      value: 0.9946666666666667
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="200" height="32"/>](https://wandb.ai/ermuzzz2001/huggingface/runs/z1fmsvrc)
# train_mae_base

This model is a fine-tuned version of [facebook/vit-mae-base](https://huggingface.co/facebook/vit-mae-base) on the imagefolder dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0154
- Accuracy: 0.9947

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
| 0.1712        | 0.1881 | 200  | 0.1031          | 0.9667   |
| 0.1097        | 0.3763 | 400  | 0.0792          | 0.973    |
| 0.1138        | 0.5644 | 600  | 0.0663          | 0.9777   |
| 0.0429        | 0.7526 | 800  | 0.0464          | 0.9817   |
| 0.0571        | 0.9407 | 1000 | 0.0585          | 0.9803   |
| 0.0604        | 1.1289 | 1200 | 0.0339          | 0.985    |
| 0.0377        | 1.3170 | 1400 | 0.0551          | 0.9827   |
| 0.0632        | 1.5052 | 1600 | 0.0289          | 0.9883   |
| 0.0235        | 1.6933 | 1800 | 0.0334          | 0.987    |
| 0.0096        | 1.8815 | 2000 | 0.0160          | 0.994    |


### Framework versions

- Transformers 4.41.0.dev0
- Pytorch 2.3.0+cu121
- Datasets 2.19.0
- Tokenizers 0.19.1
