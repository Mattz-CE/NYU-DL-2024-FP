---
license: mit
base_model: ikim-uk-essen/BiomedCLIP_ViT_patch16_224
tags:
- image-classification
- vision
- generated_from_trainer
datasets:
- imagefolder
metrics:
- accuracy
model-index:
- name: train_biomedclip_base
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
      value: 0.999
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="200" height="32"/>](https://wandb.ai/ermuzzz2001/huggingface/runs/kr78my9r)
# train_biomedclip_base

This model is a fine-tuned version of [ikim-uk-essen/BiomedCLIP_ViT_patch16_224](https://huggingface.co/ikim-uk-essen/BiomedCLIP_ViT_patch16_224) on the imagefolder dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0031
- Accuracy: 0.999

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
| 0.1378        | 0.1881 | 200  | 0.1167          | 0.967    |
| 0.0796        | 0.3763 | 400  | 0.0596          | 0.981    |
| 0.0867        | 0.5644 | 600  | 0.0605          | 0.981    |
| 0.0294        | 0.7526 | 800  | 0.0455          | 0.9853   |
| 0.0797        | 0.9407 | 1000 | 0.0255          | 0.991    |
| 0.0521        | 1.1289 | 1200 | 0.0104          | 0.996    |
| 0.0935        | 1.3170 | 1400 | 0.0072          | 0.997    |
| 0.0456        | 1.5052 | 1600 | 0.0055          | 0.9973   |
| 0.0278        | 1.6933 | 1800 | 0.0057          | 0.998    |
| 0.0068        | 1.8815 | 2000 | 0.0035          | 0.9987   |


### Framework versions

- Transformers 4.41.0.dev0
- Pytorch 2.3.0+cu121
- Datasets 2.19.0
- Tokenizers 0.19.1
