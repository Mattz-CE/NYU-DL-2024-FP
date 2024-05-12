---
license: apache-2.0
base_model: microsoft/beit-base-patch16-224
tags:
- image-classification
- vision
- generated_from_trainer
datasets:
- imagefolder
metrics:
- accuracy
model-index:
- name: train_mim_base_linear_prob
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
      value: 0.942
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="200" height="32"/>](https://wandb.ai/ermuzzz2001/huggingface/runs/eodbwykc)
# train_mim_base_linear_prob

This model is a fine-tuned version of [microsoft/beit-base-patch16-224](https://huggingface.co/microsoft/beit-base-patch16-224) on the imagefolder dataset.
It achieves the following results on the evaluation set:
- Loss: 0.4403
- Accuracy: 0.942

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
| 1.3527        | 0.1881 | 200  | 1.4179          | 0.4217   |
| 1.1308        | 0.3763 | 400  | 1.0567          | 0.7667   |
| 0.89          | 0.5644 | 600  | 0.8368          | 0.8587   |
| 0.7059        | 0.7526 | 800  | 0.6982          | 0.8903   |
| 0.646         | 0.9407 | 1000 | 0.6057          | 0.9123   |
| 0.6099        | 1.1289 | 1200 | 0.5408          | 0.9297   |
| 0.6072        | 1.3170 | 1400 | 0.4983          | 0.9317   |
| 0.5364        | 1.5052 | 1600 | 0.4695          | 0.9377   |
| 0.5311        | 1.6933 | 1800 | 0.4511          | 0.9407   |
| 0.5245        | 1.8815 | 2000 | 0.4419          | 0.9413   |


### Framework versions

- Transformers 4.41.0.dev0
- Pytorch 2.3.0+cu121
- Datasets 2.19.0
- Tokenizers 0.19.1
