# EigenHunt
RL:
```
CUDA_VISIBLE_DEVICES= python train.py --algo tqc --env BertTrotSpeed-v1 --env-kwargs action_repeat:2 --eval-freq -1 -tb logs/tb/bert-trot --save-freq 10000 --save-replay-buffer

CUDA_VISIBLE_DEVICES= python enjoy.py --algo tqc --env BertTrotSpeed-v1 -f logs/ --exp-id 0 -n 1500 --env-kwargs action_repeat:1

CUDA_VISIBLE_DEVICES= python enjoy.py --algo tqc --env BertPronk-v1 -f logs/ --exp-id 0 -n 1500 --env-kwargs action_repeat:1
```
