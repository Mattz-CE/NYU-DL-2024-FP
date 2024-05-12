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
- name: train_mae_base_linear_prob
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
      value: 0.8613333333333333
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="200" height="32"/>](https://wandb.ai/ermuzzz2001/huggingface/runs/4d47bt0v)
# train_mae_base_linear_prob

This model is a fine-tuned version of [facebook/vit-mae-base](https://huggingface.co/facebook/vit-mae-base) on the imagefolder dataset.
It achieves the following results on the evaluation set:
- Loss: 1.3886
- Accuracy: 0.8613

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
| 1.5742        | 0.1881 | 200  | 1.5786          | 0.2077   |
| 1.5468        | 0.3763 | 400  | 1.5330          | 0.37     |
| 1.4997        | 0.5644 | 600  | 1.4991          | 0.662    |
| 1.475         | 0.7526 | 800  | 1.4707          | 0.8103   |
| 1.4531        | 0.9407 | 1000 | 1.4468          | 0.8307   |
| 1.4355        | 1.1289 | 1200 | 1.4280          | 0.869    |
| 1.4091        | 1.3170 | 1400 | 1.4131          | 0.857    |
| 1.4085        | 1.5052 | 1600 | 1.4014          | 0.838    |
| 1.3928        | 1.6933 | 1800 | 1.3935          | 0.8573   |
| 1.3973        | 1.8815 | 2000 | 1.3893          | 0.8627   |


### Framework versions

- Transformers 4.41.0.dev0
- Pytorch 2.3.0+cu121
- Datasets 2.19.0
- Tokenizers 0.19.1
