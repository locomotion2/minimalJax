# Results CPG + RL


## Jump in place

Hand tuned CPG + RL

On 5 episodes, reset in the middle of the treadmill everytime
reward function commit: 14ee5edab30c072438afc57256b39f943120ed2b

## Daniel params CPG only
(action repeat = 1, 300 steps, 2.5s)
5 Episodes
Mean reward: 997.51 +/- 313.57
Mean episode length: 263.60 +/- 72.80

Note: 1 fall


## Daniel params CPG + RL
5 Episodes
Mean reward: 1490.06 +/- 171.24
Mean episode length: 300.00 +/- 0.00

Note: RL not trained with that CPG controller

### CPG only (antonin hand-tuned)

(action repeat = 1, 300 steps, 2.5s)
Mean reward: 1353.31 +/- 390.97
Mean episode length: 284.50 +/- 26.85
Note: 1 fall

## CPG optimized
5 Episodes
Mean reward: 1252.29 +/- 274.75
Mean episode length: 300.00 +/- 0.00


### CPG + RL (1h):

(action repeat = 1, 300 steps, 2.5s)
Mean reward: 1623.83 +/- 153.94
Mean episode length: 300.00 +/- 0.00

### CPG (optimized) + RL (1h):

5 Episodes
Mean reward: 1542.69 +/- 171.17
Mean episode length: 282.40 +/- 22.85

Note: off-limits twice
RL with not trained with this optimized CPG
Mean reward without: 1639


TODO: evaluate fall rate (when manually moving it), or time to fall (on the flat ground)

Evaluation
Same as during training:
```
FORCE_RESET=True CUDA_VISIBLE_DEVICES= python enjoy.py --algo tqc --env BertPronk-v1 -f logs/ --exp-id 4 -n 1500 --env-kwargs action_repeat:2
```


Note: if action repeat is set to 1 (60Hz), max episode steps need to be adjusted
```
CUDA_VISIBLE_DEVICES= python enjoy.py --algo tqc --env BertPronk-v1 -f logs/ --exp-id 4 -n 20000 --env-kwargs action_repeat:1
```

CPG only:
```
CUDA_VISIBLE_DEVICES= python enjoy.py --algo tqc --env BertPronk-v1 -f logs/ --exp-id 0 -n 20000 --env-kwargs action_repeat:2 max_offset:0.0
```

(outside tracking)
```
TRACKING_DISABLED=True CUDA_VISIBLE_DEVICES= python enjoy.py --algo tqc --env BertPronk-v1 -f logs/ --exp-id 4 -n 20000 --env-kwargs action_repeat:1
```

Train:
```
CUDA_VISIBLE_DEVICES= python train.py --algo tqc --env BertPronk-v1 -f logs/ --env-kwargs action_repeat:2 max_offset:0.004 offset_axes:both symmetry:None --eval-freq -1 --save-freq 1000 --save-replay-buffer
```

Hyperparams:
```yaml
BertPronk-v1: &defaults
  env_wrapper:
    - utils.wrappers.HistoryWrapper:
        horizon: 2
  callback:
    - utils.callbacks.ParallelTrainCallback:
        gradient_steps: 200
  n_timesteps: !!float 2e6
  policy: 'MlpPolicy'
  learning_rate: !!float 7.3e-4
  buffer_size: 200000
  batch_size: 256
  ent_coef: 'auto'
  gamma: 0.99
  tau: 0.02
  train_freq: 200
  # Synchronous training
  # train_freq: [1, "episode"]
  # gradient_steps: -1
  learning_starts: 500
  use_sde_at_warmup: True
  use_sde: True
  sde_sample_freq: 16
  policy_kwargs: "dict(log_std_init=-3, net_arch=[256, 256], n_critics=2, use_expln=True)"

```
