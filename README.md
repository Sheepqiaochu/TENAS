## Changed block
### 1.Dataloader
* Load state_dict of UAP generator and generate UAP.
* Add UAPS to origin pictures and return x and $x^\prime$

### 2.Calculate NTK score
* Calculate NTK score for both x and $x^\prime$
* Calculate weighted sum of NTK(x) and NTK$\left(x^\prime\right)$
  

## Usage
### 0. Prepare the dataset
* Please follow the guideline [here](https://github.com/D-X-Y/AutoDL-Projects#requirements-and-preparation) to prepare the CIFAR-10/100 and ImageNet dataset, and also the NAS-Bench-201 database.
* **Remember to properly set the `TORCH_HOME` and `data_paths` in the `prune_launch.py`.**

### 1. Search
#### [NAS-Bench-201 Space](https://openreview.net/forum?id=HJxyZkBKDr)
```python
python prune_launch.py --space nas-bench-201 --dataset cifar10 --UAP_info 32resnet152 --gpu 0
python prune_launch.py --space nas-bench-201 --dataset cifar100 --UAP_info 32resnet152 --gpu 0
python prune_launch.py --space nas-bench-201 --dataset ImageNet16-120 --UAP_info 32 --gpu 0
python prune_launch.py --space nas-bench-201 --dataset imagenet-1k --UAP_info 32resnet152 --gpu 0

```

#### [DARTS Space](https://openreview.net/forum?id=S1eYHoC5FX) ([NASNET](https://openaccess.thecvf.com/content_cvpr_2018/html/Zoph_Learning_Transferable_Architectures_CVPR_2018_paper.html))
```python
python prune_launch.py --space darts --dataset cifar10 --gpu 0
python prune_launch.py --space darts --dataset imagenet-1k --gpu 0
```

### 2. Evaluation
* For architectures searched on `nas-bench-201`, the accuracies are immediately available at the end of search (from the console output).
* For architectures searched on `darts`, please use [DARTS_evaluation](https://github.com/chenwydj/DARTS_evaluation) for training the searched architecture from scratch and evaluation.


