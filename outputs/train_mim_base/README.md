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
- name: train_mim_base
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
      value: 0.9986666666666667
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="200" height="32"/>](https://wandb.ai/ermuzzz2001/huggingface/runs/53a415lk)
# train_mim_base

This model is a fine-tuned version of [microsoft/beit-base-patch16-224](https://huggingface.co/microsoft/beit-base-patch16-224) on the imagefolder dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0028
- Accuracy: 0.9987

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
| 0.0991        | 0.1881 | 200  | 0.0947          | 0.975    |
| 0.1004        | 0.3763 | 400  | 0.0613          | 0.981    |
| 0.0658        | 0.5644 | 600  | 0.0885          | 0.9763   |
| 0.0806        | 0.7526 | 800  | 0.0738          | 0.975    |
| 0.0474        | 0.9407 | 1000 | 0.0189          | 0.9927   |
| 0.068         | 1.1289 | 1200 | 0.0171          | 0.995    |
| 0.0552        | 1.3170 | 1400 | 0.0143          | 0.9967   |
| 0.0533        | 1.5052 | 1600 | 0.0093          | 0.9973   |
| 0.0344        | 1.6933 | 1800 | 0.0092          | 0.9967   |
| 0.0264        | 1.8815 | 2000 | 0.0058          | 0.9977   |


### Framework versions

- Transformers 4.41.0.dev0
- Pytorch 2.3.0+cu121
- Datasets 2.19.0
- Tokenizers 0.19.1
