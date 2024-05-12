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
- name: train_vit_huge_linear_prob
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
      value: 0.8433333333333334
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="200" height="32"/>](https://wandb.ai/ermuzzz2001/huggingface/runs/79os8t9o)
# train_vit_huge_linear_prob

This model is a fine-tuned version of [google/vit-huge-patch14-224-in21k](https://huggingface.co/google/vit-huge-patch14-224-in21k) on the imagefolder dataset.
It achieves the following results on the evaluation set:
- Loss: 1.3358
- Accuracy: 0.8433

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
| 1.5795        | 0.1881 | 200  | 1.5646          | 0.5623   |
| 1.536         | 0.3763 | 400  | 1.5153          | 0.716    |
| 1.4806        | 0.5644 | 600  | 1.4736          | 0.7163   |
| 1.4555        | 0.7526 | 800  | 1.4378          | 0.7717   |
| 1.4436        | 0.9407 | 1000 | 1.4084          | 0.7727   |
| 1.4137        | 1.1289 | 1200 | 1.3847          | 0.8167   |
| 1.4003        | 1.3170 | 1400 | 1.3655          | 0.838    |
| 1.3872        | 1.5052 | 1600 | 1.3515          | 0.8277   |
| 1.3657        | 1.6933 | 1800 | 1.3417          | 0.8377   |
| 1.3886        | 1.8815 | 2000 | 1.3367          | 0.8433   |


### Framework versions

- Transformers 4.41.0.dev0
- Pytorch 2.3.0+cu121
- Datasets 2.19.0
- Tokenizers 0.19.1
