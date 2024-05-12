---
license: apache-2.0
base_model: google/vit-base-patch16-224-in21k
tags:
- image-classification
- vision
- generated_from_trainer
datasets:
- imagefolder
metrics:
- accuracy
model-index:
- name: train_vit_base
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

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="200" height="32"/>](https://wandb.ai/ermuzzz2001/huggingface/runs/jy4cs7kb)
# train_vit_base

This model is a fine-tuned version of [google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k) on the imagefolder dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0323
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
| 0.2286        | 0.1881 | 200  | 0.1944          | 0.9763   |
| 0.1288        | 0.3763 | 400  | 0.1368          | 0.9797   |
| 0.0888        | 0.5644 | 600  | 0.1014          | 0.9853   |
| 0.0973        | 0.7526 | 800  | 0.0999          | 0.9803   |
| 0.0818        | 0.9407 | 1000 | 0.0891          | 0.9807   |
| 0.0948        | 1.1289 | 1200 | 0.0532          | 0.9923   |
| 0.0617        | 1.3170 | 1400 | 0.0502          | 0.9927   |
| 0.1005        | 1.5052 | 1600 | 0.0451          | 0.992    |
| 0.0481        | 1.6933 | 1800 | 0.0356          | 0.9957   |
| 0.052         | 1.8815 | 2000 | 0.0344          | 0.9957   |


### Framework versions

- Transformers 4.41.0.dev0
- Pytorch 2.3.0+cu121
- Datasets 2.19.0
- Tokenizers 0.19.1
