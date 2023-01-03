# CPG Pronk

## hand tuned

|                               | CPG (pronk)      |
|-------------------------------|------------------|
| Mean reward/step              |  5.3             |
| Mean reward                   |  1274 +/- 396    |
| Mean Speed (m/s)              |  0.05 +/- 0.11   |
| Mean cost of transport (elec) |  0.29 +/- 0.02   |
| Mean CoT (mecha)              |  0.038 +/- 0.003 |


## Daniel hand-tuned

|                               | CPG (pronk)      |
|-------------------------------|------------------|
| Mean reward/step              |               |
| Mean reward                   |     |
| Mean Speed (m/s)              |   +/- 0.11   |
| Mean cost of transport (elec) |   +/- 0.02   |
| Mean CoT (mecha)              |  +/- 0.003 |

5 Episodes
Mean reward: 869.03 +/- 346.46
Mean episode length: 244.60 +/- 95.03
1 failure


## Optimized for reward/step

|                               | CPG (pronk)      |
|-------------------------------|------------------|
| Mean reward/step              |  6.07            |
| Mean reward                   |  1615 +/- 300    |
| Mean Speed (m/s)              |  0.15 +/- 0.03   |
| Mean cost of transport (elec) |  0.25 +/- 0.032  |
| Mean CoT (mecha)              |  0.034 +/- 0.005 |
| Improvement (reward)          | 20% better       |

6 episodes
Mean reward: 928.11 +/- 229.41
Mean episode length: 218.00 +/- 51.86
1 failure + off the tracking limits

With RL:
6 Episodes
Mean reward: 1381.70 +/- 417.43
Mean episode length: 237.00 +/- 68.21
Note: off the tracking limits once
mean reward without: 1765


TODO: compute max height / max z velocity

Notes:

- using `python optim/eval_params.py --storage sqlite:///logs/studies.db --env CPGBertPronk-v1 -name cpg-optimize-pronk -trial 135 --n-eval-episodes 3 --max-episode-steps 300`


## Plots

On flat ground (no tracking)
```
python3 bert_utils/visualization/plot_spring.py -i logs/spring_data_1000_Hz_pronk_cpg_1_4.0s_1661856592.npz

python3 bert_utils/visualization/plot_spring.py -i logs/spring_data_1000_Hz_pronk_rl_1_4.0s_1661850034.npz

python3 bert_utils/visualization/plot_spring.py -i logs/spring_data_1000_Hz_pronk_daniel_1_4.0s_1662047291.npz

python3 bert_utils/visualization/plot_spring.py -i logs/spring_data_1000_Hz_pronk_optimized_gamepad_1_4.0s_1662048676.npz

```

On the treadmill:
```
python3 bert_utils/visualization/plot_spring.py -i logs/spring_data_60_Hz_pronk_cpg_2_4.0s_1661181076.npz --with-tracker

python3 bert_utils/visualization/plot_spring.py -i logs/spring_data_60_Hz_pronk_rl_2_4.0s_1661180973.npz --with-tracker

python3 bert_utils/visualization/plot_spring.py -i logs/spring_data_60_Hz_pronk_optimized_gamepad_2_4.0s_1662048181.npz --with-tracker

python3 bert_utils/visualization/plot_spring.py -i logs/spring_data_60_Hz_pronk_daniel_1_4.0s_1662047997.npz --with-tracker
```

Plot paper:
```
python3 bert_utils/visualization/plot_spring.py -i logs/spring_data_1000_Hz_pronk_daniel_1_4.0s_1662047291.npz -s 1000 -e 3000

python3 bert_utils/visualization/plot_spring.py -i logs/spring_data_1000_Hz_pronk_optimized_gamepad_1_4.0s_1662048676.npz -s 1000 -e 3000

python3 bert_utils/visualization/plot_spring.py -i logs/spring_data_60_Hz_pronk_daniel_1_4.0s_1662047997.npz --with-tracker -s 120 -e 240

python3 bert_utils/visualization/plot_spring.py -i logs/spring_data_60_Hz_pronk_optimized_gamepad_2_4.0s_1662048181.npz --with-tracker -s 100 -e 220

python3 bert_utils/visualization/plot_spring.py -i logs/spring_data_60_Hz_pronk_rl_2_4.0s_1661180973.npz --with-tracker -s 120 -e 240
```
