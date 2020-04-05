# text-subgan



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