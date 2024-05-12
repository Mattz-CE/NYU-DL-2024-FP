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
- name: train_clip_large_linear_prob
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
      value: 0.845
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="200" height="32"/>](https://wandb.ai/ermuzzz2001/huggingface/runs/u391cupw)
# train_clip_large_linear_prob

This model is a fine-tuned version of [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) on the imagefolder dataset.
It achieves the following results on the evaluation set:
- Loss: 1.1327
- Accuracy: 0.845

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
| 1.5133        | 0.1881 | 200  | 1.5163          | 0.3877   |
| 1.4472        | 0.3763 | 400  | 1.4255          | 0.715    |
| 1.3541        | 0.5644 | 600  | 1.3538          | 0.7607   |
| 1.3105        | 0.7526 | 800  | 1.2942          | 0.7963   |
| 1.2776        | 0.9407 | 1000 | 1.2456          | 0.7917   |
| 1.2284        | 1.1289 | 1200 | 1.2079          | 0.8417   |
| 1.2052        | 1.3170 | 1400 | 1.1790          | 0.8373   |
| 1.1748        | 1.5052 | 1600 | 1.1565          | 0.832    |
| 1.1563        | 1.6933 | 1800 | 1.1417          | 0.8433   |
| 1.1868        | 1.8815 | 2000 | 1.1341          | 0.8453   |


### Framework versions

- Transformers 4.41.0.dev0
- Pytorch 2.3.0+cu121
- Datasets 2.19.0
- Tokenizers 0.19.1
