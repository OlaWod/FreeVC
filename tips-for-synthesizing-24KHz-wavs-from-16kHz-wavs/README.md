# tips for synthesizing 24KHz wavs from 16kHz wavs

It is easy to synthesise wavs from 16kHz inputs at another sampling rate (e.g. 24kHz), just set a proper hop length and modify the `upsample_rates` in decoder.

> For example, 
> input wav: sampling_rate=16kHz, hop_length=320; 
> output wav: sampling_rate=24kHz, hop_length=240. 
> Then we just need to set `upsample_rates` to 480.

This directory provides the code I used to train a model that outputs 24kHz wavs. The parameters are hardcoded because I am lazy. If this attracts enough interest I might consider sort it out.  The pretrained checkpoint (freevc-24.pth) is also provided in the '24kHz' dir [here](https://1drv.ms/u/s!AnvukVnlQ3ZTx1rjrOZ2abCwuBAh?e=UlhRR5).

## Inference Example

```python
CUDA_VISIBLE_DEVICES=0 python convert_24.py --hpfile logs/freevc-24.json --ptfile checkpoints/freevc-24.pth --txtpath convert.txt --outdir outputs/freevc-24
```

## Training Example

```python
CUDA_VISIBLE_DEVICES=0 python train_24.py -c configs/freevc-24.json -m freevc-24
```
