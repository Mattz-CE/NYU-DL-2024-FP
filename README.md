# DONE and TODO:
DONE:
1. Finetuning a google/vit-base-patch16-224-in21k

# Install
```
conda create -n dlp python=3.10
```

# Download data
```
wget https://huggingface.co/datasets/ermu2001/LungColonCancerClassification/resolve/main/lung_colon_dataset.zip
unzip lung_colon_dataset.zip
```

# Train
```
bash train.sh
```

# Explores data and Baseline Model Training
Use jupyter notebook on cnn_train.ipynb