# graph-reconstruction-amr2text
An implementation for paper "Better AMR-To-Text Generation with Graph Structure Reconstruction" (accepted at IJCAI20)
(coming soon)

## Requirements
* python 3.5
* tensorflow 1.5

## Data Preprocessing
We use the tools in https://github.com/Cartus/DCGCN (without anonymization) to preprocessing the data. Because AMR corpus has LDC license, we cannot distribute the preprocessed data. We upload some examples for format reference. If you have the license, feel free to contact us for getting the preprocessed data.

We use pretrained Glove vectors. Due to the limitation of filesize, we only upload part of pretrained vectors. You can extract it from "glove.840B.300d.txt" with the vocab file.

## Train
```
python train.py --enc_layers=6 --dec_layers=6 --num_heads=2 --num_units=512 --emb_dim=300  --train_dir=ckpt/ --use_copy=1 --batch_size=64 --dropout_rate=0.3 --gpu_device=0 --max_src_len=50 --max_tgt_len=50
```

## Test
```
python infer.py --enc_layers=6 --dec_layers=6 --num_heads=2 --num_units=512 --emb_dim=300  --train_dir=ckpt/ --use_copy=1 --batch_size=64 --dropout_rate=0.3 --gpu_device=0 --max_src_len=50 --max_tgt_len=50
```
#The output file can be found in the  folder directory.
