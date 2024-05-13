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
- name: train_clip_base_linear_prob
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
      value: 0.7443333333333333
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="200" height="32"/>](https://wandb.ai/ermuzzz2001/huggingface/runs/9o6gswwv)
# train_clip_base_linear_prob

This model is a fine-tuned version of [openai/clip-vit-base-patch16](https://huggingface.co/openai/clip-vit-base-patch16) on the imagefolder dataset.
It achieves the following results on the evaluation set:
- Loss: 1.2941
- Accuracy: 0.7443

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
| 1.5449        | 0.1881 | 200  | 1.5449          | 0.364    |
| 1.5022        | 0.3763 | 400  | 1.4907          | 0.5447   |
| 1.4481        | 0.5644 | 600  | 1.4452          | 0.5943   |
| 1.4277        | 0.7526 | 800  | 1.4061          | 0.665    |
| 1.4074        | 0.9407 | 1000 | 1.3739          | 0.6713   |
| 1.3611        | 1.1289 | 1200 | 1.3478          | 0.699    |
| 1.3484        | 1.3170 | 1400 | 1.3277          | 0.7153   |
| 1.3322        | 1.5052 | 1600 | 1.3118          | 0.714    |
| 1.3375        | 1.6933 | 1800 | 1.3007          | 0.7363   |
| 1.3346        | 1.8815 | 2000 | 1.2950          | 0.7437   |


### Framework versions

- Transformers 4.41.0.dev0
- Pytorch 2.3.0+cu121
- Datasets 2.19.0
- Tokenizers 0.19.1
