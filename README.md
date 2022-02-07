# LINDA: Unsupervised Learning to Interpolate in Natural Language Processing

Official PyTorch Implementation of SSMix | [Paper](https://arxiv.org/abs/2112.13969) 
* * *
## Abstract
Despite the success of mixup in data augmentation, its applicability to natural language processing (NLP) tasks has been limited due to the discrete and variable-length nature of natural languages. Recent studies have thus relied on domain-specific heuristics and manually crafted resources, such as dictionaries, in order to apply mixup in NLP. In this paper, we instead propose an unsupervised learning approach to text interpolation for the purpose of data augmentation, to which we refer as "Learning to INterpolate for Data Augmentation" (LINDA), that does not require any heuristics nor manually crafted resources but learns to interpolate between any pair of natural language sentences over a natural language manifold. After empirically demonstrating the LINDA's interpolation capability, we show that LINDA indeed allows us to seamlessly apply mixup in NLP and leads to better generalization in text classification both in-domain and out-of-domain.

## Installation

```
conda create -n linda python==3.7
conda activate linda
pip install -r requirements.txt
cd LINDA
```

## Datasets
> Dataset path: `./LINDA/data/raw`
* `train.txt`: train set of 1,000,000 sentences (Currently not provided because of size issues)
* `train_tiny.txt`: train set of 100 sentences
* `dev.txt`: dev set of 50,000 sentences

### Training with a specific train set
1. set path to the train set to be used in `dataset_loading_script.py`
```python
# when using train_tiny.txt
TRAIN_DATA_PATH = os.path.join(data_dir, "train_tiny.txt") 
#when using train.txt
TRAIN_DATA_PATH = os.path.join(data_dir, "train.txt")
```
2. set train_mode arguments in `run.py`
```python
# either 'full' or 'tiny'
parser.add_argument("--train_mode", type=str, default='tiny',
                        help="'full' or 'tiny'. If set to 'tiny', trained with only 100 sentence pairs.")
```

## Important Arguments
* `--gpu_ids`: specify gpu ids
* `--train_mode`: set accordingly to the path set in `dataset_loading_script.py`

## Example
```
python3 LINDA/run.py --do_train --train_mode tiny --per_device_train_batch_size 16 --gpu_ids 0,1,2,3,4,5,6,7
```
***

