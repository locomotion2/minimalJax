# EigenHunt

How to train the model:
```
python train.py --algo tqc --env TestEnvironment-v1 -P --conf-file hyperparams/tqc.yml --env-kwargs render:False mode:"'speed'" --eval-freq -1 -params train_freq:10 gradient_steps:10 --log-interval 100 -n 200000
```
How to run a trained model:
```
python enjoy.py --algo tqc --env TestEnvironment-v1  --env-kwargs render:True mode:"'speed'" solve:False energy_command:0.3 -f logs/ -n 100
```
How to plot the learning curve:
```
python plot_train.py --algo tqc --env TestEnvironment-v1 -f logs
```