---
base_model: scratch
tags:
- image-classification
- vision
- generated_from_trainer
datasets:
- imagefolder
metrics:
- accuracy
model-index:
- name: train_scratch_vitbase
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
      value: 0.8553333333333333
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="200" height="32"/>](https://wandb.ai/ermuzzz2001/huggingface/runs/41tc1rcj)
# train_scratch_vitbase

This model is a fine-tuned version of [scratch](https://huggingface.co/scratch) on the imagefolder dataset.
It achieves the following results on the evaluation set:
- Loss: 0.3282
- Accuracy: 0.8553

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.001
- train_batch_size: 64
- eval_batch_size: 16
- seed: 1337
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 20.0

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| 0.5541        | 1.0   | 266  | 0.6526          | 0.7277   |
| 0.4421        | 2.0   | 532  | 0.5132          | 0.7707   |
| 0.4562        | 3.0   | 798  | 0.5665          | 0.7427   |
| 0.4693        | 4.0   | 1064 | 0.4409          | 0.8003   |
| 0.4551        | 5.0   | 1330 | 0.4940          | 0.7873   |
| 0.4238        | 6.0   | 1596 | 0.4784          | 0.809    |
| 0.4255        | 7.0   | 1862 | 0.4132          | 0.8113   |
| 0.4043        | 8.0   | 2128 | 0.4161          | 0.801    |
| 0.428         | 9.0   | 2394 | 0.3967          | 0.825    |
| 0.381         | 10.0  | 2660 | 0.3669          | 0.841    |
| 0.3683        | 11.0  | 2926 | 0.3726          | 0.8343   |
| 0.4106        | 12.0  | 3192 | 0.3743          | 0.8307   |
| 0.3801        | 13.0  | 3458 | 0.3932          | 0.8353   |
| 0.3618        | 14.0  | 3724 | 0.3849          | 0.828    |
| 0.3616        | 15.0  | 3990 | 0.3612          | 0.8363   |
| 0.3598        | 16.0  | 4256 | 0.3545          | 0.84     |
| 0.3527        | 17.0  | 4522 | 0.3494          | 0.8433   |
| 0.299         | 18.0  | 4788 | 0.3302          | 0.857    |
| 0.2926        | 19.0  | 5054 | 0.3266          | 0.8583   |
| 0.318         | 20.0  | 5320 | 0.3282          | 0.8553   |


### Framework versions

- Transformers 4.41.0.dev0
- Pytorch 2.3.0+cu121
- Datasets 2.19.0
- Tokenizers 0.19.1
