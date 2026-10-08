# HashUQ
Implementation of paper: Hashing with Uncertainty Quantification via Sampling-based Hypothesis Testing. We implement our model based on [DeepHash-pytorch](https://github.com/swuxyj/DeepHash-pytorch).

# Environment

Current setup (Python 3.11, torch 2.x, CUDA 11.8):

```
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python check_env.py --dataset imagenet   # checks packages, GPU, AlexNet weights, dataset files
```

Original environment from the paper:

```
torch==1.8.1+cu111
lap==0.4.0
matplotlib==3.4.2
torchvision==0.9.1+cu111
scipy==1.6.3
numpy==1.19.5
tqdm==4.61.0
pandas==1.2.4
Flask==3.0.0
Pillow==10.1.0
scikit_learn==1.3.1
```

# Dataset Preparation
Please find the instructions to setup datasets in [DeepHash-pytorch](https://github.com/swuxyj/DeepHash-pytorch). Notice that the specfic split we use for NUS-WIDE dataset is NUS-WIDE-21. 
After downloading the dataset, please unzip the dataset at
```
../image_hashing_data/DATA_SET
```

where DATA_SET represent the specfic dataset we want to test, and need to be replaced by ``imagenet``, ``coco`` or ``nuswide_21``.

We include our dataset split in ```./data/```. The split is constructed by randomly select 10% of the data samples in the original training split to do the validation and model selection using ```Devide_TrainingSet_ImageNet```, ```Devide_TrainingSet_MSCOCO``` and ```Devide_TrainingSet_NUSWIDE```. Please move the txt files to the root folder of each dataset. 


# Training
To train our model with ``Center-Target'' construction on ImageNet, use the command:
```
python CSQ_HashUQ.py --grad_est cf --no-pairwise --no-tqdm --dataset imagenet --KL_regularization 1 --val --rt_st hamuct --sample_method MCD --sample 100
```

You can replace the argument ``imagenet`` to ``coco`` or ``nuswide_21`` to train and check the performance on other dataset. 

Extra arguments (both `CSQ_HashUQ.py` and `GreedyHash_HashUQ.py`):

| Argument | Default | Meaning |
|---|---|---|
| `--bit 16 32 64` | `16` | Hash lengths to train, one after another |
| `--seed N` | `0` | Random seed (also part of the save folder name, so runs with different seeds don't overwrite each other) |
| `--net NAME` | `alexnet` | `alexnet`, `resnet18`, `resnet34`, `resnet50`, ... (ResNet has no Dropout layer, so MC Dropout gives identical samples) |
| `--epoch N` | `100` | Training epochs (validation runs every 10 epochs for CSQ, every 100 for GreedyHash) |
| `--num_workers N` | `4` | DataLoader workers (use 0–4 on Windows/laptops) |
| `--uct_quant_levels 2 4` | `2 4` | HashUQ-Bin quantization levels evaluated together with HashUQ |

With `--rt_st hamuct`, each evaluation prints and saves Hamming, HashUQ and HashUQ-Bin mAP to
`saved_model/<run>/results_epoch_<N>.json`; the best-checkpoint evaluation is saved as `results_final.json`.
The database/query uncertainties are saved as `trn_uct.npy` / `tst_uct.npy` next to the hash codes.

Model will be automaticly saved at 
```
./saved_model/
```

# Testing
To test our model with ``Center-Target'' construction on ImageNet after trai, use the command:
```
python CSQ_HashUQ.py --grad_est cf --no-pairwise --no-tqdm --dataset imagenet --KL_regularization 1 --val --rt_st hamuct --sample_method MCD --sample 100 --no-train
```