#### MFTS

Official Code for “Leveraging Sound Source Trajectories for Universal Sound Separation”, available at https://arxiv.org/abs/2409.04843

##### Dataset generation

```
cd ./simulation
python simulate_rir_rb_angle_diff_cross_fixbug.py
```



##### Training

envelope estimation

```
python -W ignore -m src.training.train.train_trans_env experiments/trans_env --use_cuda
```

Initial tracking

```
python -W ignore -m src.training.train.train_trans_locinit experiments/trans_initloc --use_cuda
```

target sound extraction

```
python -W ignore -m src.training.train.train_trans_tse experiments/trans_STIV --use_cuda
```

precise tracking

```
python -W ignore -m src.training.train.train_trans_locprec experiments/trans_precloc --use_cuda
```

neural beamforming

```
python -W ignore -m src.training.train.train_trans_bf_envfirst experiments/trans_bf --use_cuda
```

###### training baselines

convtasnet, fasnettac, spatialnet

```
python -W ignore -m src.training.train.train_trans_noinfor experiments/trans_noinfor --use_cuda
```

mix-track

```
python -W ignore -m src.training.train.train_trans_locnoinfor experiments/trans_noinforloc --use_cuda
```





This code is adapted from the original open source repository at 

- https://github.com/vb000/Waveformer
- https://github.com/Audio-WestlakeU/NBSS
- https://github.com/donghoney0416/IC_Conv-TasNet
- https://github.com/tky823/audio_source_separation
- https://github.com/espnet/espnet

Thanks for the open source.