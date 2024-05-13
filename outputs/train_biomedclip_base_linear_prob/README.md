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
- name: train_biomedclip_base_linear_prob
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
      value: 0.93
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="200" height="32"/>](https://wandb.ai/ermuzzz2001/huggingface/runs/yhq9fb9z)
# train_biomedclip_base_linear_prob

This model is a fine-tuned version of [ikim-uk-essen/BiomedCLIP_ViT_patch16_224](https://huggingface.co/ikim-uk-essen/BiomedCLIP_ViT_patch16_224) on the imagefolder dataset.
It achieves the following results on the evaluation set:
- Loss: 0.2255
- Accuracy: 0.93

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
| 0.9139        | 0.1881 | 200  | 0.7583          | 0.7787   |
| 0.6024        | 0.3763 | 400  | 0.4801          | 0.8607   |
| 0.3679        | 0.5644 | 600  | 0.3839          | 0.8883   |
| 0.3574        | 0.7526 | 800  | 0.3118          | 0.9157   |
| 0.3302        | 0.9407 | 1000 | 0.2787          | 0.92     |
| 0.302         | 1.1289 | 1200 | 0.2580          | 0.9227   |
| 0.3346        | 1.3170 | 1400 | 0.2434          | 0.928    |
| 0.2864        | 1.5052 | 1600 | 0.2354          | 0.9287   |
| 0.2471        | 1.6933 | 1800 | 0.2306          | 0.928    |
| 0.2776        | 1.8815 | 2000 | 0.2259          | 0.9307   |


### Framework versions

- Transformers 4.41.0.dev0
- Pytorch 2.3.0+cu121
- Datasets 2.19.0
- Tokenizers 0.19.1
