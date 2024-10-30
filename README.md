# FPO-CMH
The source code for the paper "Fast Partial-modal Online Cross-Modal Hashing".

## datasets & pre-trained cmh models
1. Download datasets and pre-trained scmh&ucmh models

```
url:  https://pan.baidu.com/s/1cVwg3UueZCZ7We63-IsYwg
code: 2ov3
```

2. Change the value of `root` in file `configs\config.yaml` to `/path/to/dataset`.

## pre-trained clip model
1. Download the pre-trained model from `huggingface.co`

``` bash
git lfs install
git clone https://huggingface.co/openai/clip-vit-base-patch32
```

2. Modify the value of `clip_model` in the file `configs\config.yaml`.

## python environment
``` bash
conda create -n FPO_CMH python=3.8
conda activate FPO_CMH
pip install requirements.txt
```

## training
``` python
python main.py
```
## evaluation
``` python
python eval.py
```

## Reference
...