---
license: apache-2.0
base_model: microsoft/beit-large-patch16-224
tags:
- image-classification
- vision
- generated_from_trainer
datasets:
- imagefolder
metrics:
- accuracy
model-index:
- name: train_mim_large_linear_prob
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
      value: 0.954
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="200" height="32"/>](https://wandb.ai/ermuzzz2001/huggingface/runs/qk0ouxnd)
# train_mim_large_linear_prob

This model is a fine-tuned version of [microsoft/beit-large-patch16-224](https://huggingface.co/microsoft/beit-large-patch16-224) on the imagefolder dataset.
It achieves the following results on the evaluation set:
- Loss: 0.2050
- Accuracy: 0.954

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
| 1.1498        | 0.1881 | 200  | 0.9909          | 0.7413   |
| 0.7942        | 0.3763 | 400  | 0.5965          | 0.8843   |
| 0.456         | 0.5644 | 600  | 0.4192          | 0.926    |
| 0.4125        | 0.7526 | 800  | 0.3321          | 0.9367   |
| 0.3074        | 0.9407 | 1000 | 0.2827          | 0.9423   |
| 0.3249        | 1.1289 | 1200 | 0.2503          | 0.9473   |
| 0.2955        | 1.3170 | 1400 | 0.2306          | 0.949    |
| 0.275         | 1.5052 | 1600 | 0.2183          | 0.9507   |
| 0.2866        | 1.6933 | 1800 | 0.2097          | 0.9523   |
| 0.3253        | 1.8815 | 2000 | 0.2057          | 0.9533   |


### Framework versions

- Transformers 4.41.0.dev0
- Pytorch 2.3.0+cu121
- Datasets 2.19.0
- Tokenizers 0.19.1
