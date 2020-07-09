# text-subgan

0. Setup

```
pip install -r requirements.txt
```

1. Preprocess and save data

```
python preprocess2.py  \
    -train_src dataset/train_article.txt \
    -train_tgt dataset/train_title.txt \
    -train_template dataset/train_template.txt \
    -valid_src dataset/valid_article.txt \
    -valid_tgt dataset/valid_title.txt \
    -valid_template dataset/valid_template.txt \
    -test_src dataset/test_article.txt \
    -test_tgt dataset/test_title.txt \
    -test_template dataset/test_template.txt \
    -save_data dataset/ -user_data dataset/user_data -overwrite
```

2. Train model

```
python -m cli.tempest_trainer --batch-size 48 --iterations 100000 --biset True --name biset

python -m cli.tempest_gate_trainer --batch-size 48 --iterations 100000 --name rec-gate-mf
```

3. Evaluate 

```
python -m cli.tempest_trainer --evaluate True -ckpt save/tempest_rec-2020-06-22-01-39-08/checkpoint_66000.pt
```


| BLEU      | scores |
|-----------|-------|
|BLEU-1    | 0.467 |
|BLEU-2    | 0.320 |
|BLEU-3    | 0.220 |
|BLEU-4    | 0.155 |



# Text GAN based on https://github.com/williamSYSU/TextGAN-PyTorch

1. RelGAN - RelGAN: Relational Generative Adversarial Networks for Text Generation


```
    python -m cli.relgan_trainer --pretrain-gen save/relgan_G_pretrained.pt
```


2. Sub Space RelGAN


```
    python -m cli.sub_relgan_trainer --pretrain-gen save/subspace_relgan_G_pretrained_20.pt
```



```
python -m cli.sub_relgan_trainer  --name char-no-bin-wgan --iterations 100000 --max-seq-len 50 --batch-size 128 --tokenize char --loss-type wasstestein --pretrain-gen save/subspace_relgan_G_pretrained_20_char.pt --dis-lr 0.0001 --dis-steps 7
```

### Rust Logs:

```
python -W ignore -m cli.sub_relgan_trainer  --name relgan-gp-bin-loss-0.5-kmeans --pretrain-epochs 60 --iterations 200000 --max-seq-len 40 --batch-size 64 --full-text True --grad-penalty True --pretrain-gen save/subspace_relgan_G_pretrained_20_word.pt
```

```
python -W ignore -m cli.sub_relgan_trainer  --name relgan-mini-kmeans --bin-weight -1 --pretrain-epochs 60 --iterations 200000 --max-seq-len 45 --batch-size 64 --pretrain-gen save/subspace_relgan_G_pretrained_20_word.pt
```

```
python -W ignore -m cli.sub_relgan_trainer  --name relgan-spectral --bin-weight -1 --pretrain-epochs 60 --iterations 200000 --max-seq-len 45 --batch-size 64 --pretrain-gen save/subspace_relgan_G_pretrained_20_word.pt
```


## Server Logs:

```
python -m cli.sub_relgan_trainer  --name relgan-bin-loss-0.5-kmeans --pretrain-epochs 60 --iterations 200000 --max-seq-len 40 --batch-size 64
```

```
CUDA_VISIBLE_DEVICES=1 python -m cli.sub_relgan_trainer  --name relgan-mini-kmeans --pretrain-gen ./save/subspace_relgan_G_pretrained_20_word.pt --iterations 200000 --max-seq-len 45 --batch-size 64 --bin-weight 1.0
```

```
CUDA_VISIBLE_DEVICES=3 python -m cli.sub_relgan_trainer  --name relgan-bin-loss-0.5-mini-kmeans --pretrain-epochs 100 --iterations 200000 --max-seq-len 45 --batch-size 64 --pretrain-gen ./save/subspace_relgan_G_pretrained_20_word.pt --bin-weight 0.5
```

Note:

subrelgan-bin-loss-0.5-kmeans-2020-04-07-01-33-53

Discriminator bin loss came from clusterer output


subrelgan-bin-loss-0.5-kmeans-2020-04-07-06-00-03

Discriminator bin loss came from label ( precalculated )




```
python -m cli.kkday_trainer --pretrain-embeddings ./data/kkday_dataset/model-128.bin --gen-embed-dim 128
```



