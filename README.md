# Title

## Pretrained data

You can download the CAS dataset from [this](https://doi.org/10.1038/s41597-023-02847-z).

## Requirments

1. pytorch >= 2.0
2. rasterio >= 1.3.9
3. scikit-learn >= 1.41

## Pretrain a model

  `python train.py`
  
The settings can be changed in ```train.py```:
```
if __name__ == "__main__":
    #  settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1

    train_root = "the path to CAS" 
    val_root = "the path to a dataset" 
    model_name = "AMGUnet"
    optim_params = dict(lr=0.01, momentum=0.9, weight_decay=0.001)
    epoch_set = 2
    resume = None
    save_best = True
```

The model's weights file will be in `./save_weights`

# Citation

# 
