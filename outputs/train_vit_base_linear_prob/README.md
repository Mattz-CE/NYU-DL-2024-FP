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
- name: train_vit_base_linear_prob
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
      value: 0.804
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="200" height="32"/>](https://wandb.ai/ermuzzz2001/huggingface/runs/9x4ppx36)
# train_vit_base_linear_prob

This model is a fine-tuned version of [google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k) on the imagefolder dataset.
It achieves the following results on the evaluation set:
- Loss: 1.2385
- Accuracy: 0.804

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
| 1.5432        | 0.1881 | 200  | 1.5470          | 0.4557   |
| 1.4911        | 0.3763 | 400  | 1.4767          | 0.6727   |
| 1.4238        | 0.5644 | 600  | 1.4195          | 0.7223   |
| 1.3821        | 0.7526 | 800  | 1.3714          | 0.763    |
| 1.349         | 0.9407 | 1000 | 1.3319          | 0.7757   |
| 1.3208        | 1.1289 | 1200 | 1.3010          | 0.7933   |
| 1.3046        | 1.3170 | 1400 | 1.2766          | 0.8043   |
| 1.2966        | 1.5052 | 1600 | 1.2585          | 0.8      |
| 1.2481        | 1.6933 | 1800 | 1.2461          | 0.802    |
| 1.2799        | 1.8815 | 2000 | 1.2396          | 0.8037   |


### Framework versions

- Transformers 4.41.0.dev0
- Pytorch 2.3.0+cu121
- Datasets 2.19.0
- Tokenizers 0.19.1
