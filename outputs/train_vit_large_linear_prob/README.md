---
license: apache-2.0
base_model: google/vit-large-patch16-224-in21k
tags:
- image-classification
- vision
- generated_from_trainer
datasets:
- imagefolder
metrics:
- accuracy
model-index:
- name: train_vit_large_linear_prob
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
      value: 0.9126666666666666
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="200" height="32"/>](https://wandb.ai/ermuzzz2001/huggingface/runs/ytzptpc6)
# train_vit_large_linear_prob

This model is a fine-tuned version of [google/vit-large-patch16-224-in21k](https://huggingface.co/google/vit-large-patch16-224-in21k) on the imagefolder dataset.
It achieves the following results on the evaluation set:
- Loss: 0.7725
- Accuracy: 0.9127

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
| 1.3943        | 0.1881 | 200  | 1.3988          | 0.686    |
| 1.2335        | 0.3763 | 400  | 1.2248          | 0.8383   |
| 1.0916        | 0.5644 | 600  | 1.0953          | 0.871    |
| 0.9911        | 0.7526 | 800  | 0.9981          | 0.892    |
| 0.9466        | 0.9407 | 1000 | 0.9265          | 0.8927   |
| 0.8817        | 1.1289 | 1200 | 0.8732          | 0.902    |
| 0.84          | 1.3170 | 1400 | 0.8329          | 0.909    |
| 0.8213        | 1.5052 | 1600 | 0.8037          | 0.908    |
| 0.7921        | 1.6933 | 1800 | 0.7839          | 0.9113   |
| 0.8157        | 1.8815 | 2000 | 0.7741          | 0.9127   |


### Framework versions

- Transformers 4.41.0.dev0
- Pytorch 2.3.0+cu121
- Datasets 2.19.0
- Tokenizers 0.19.1
