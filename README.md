# EigenHunt

How to train the model:
```
python -m rl_zoo3.train --algo tqc --env TestEnvironment-v1 --progress --conf-file .\hyperparams\tqc.yml --env-kwargs render:False --eval-freq 600
```
How to run a trained model:
```
python -m rl_zoo3.enjoy --algo tqc --env TestEnvironment-v1 --env-kwargs render:True --f .\logs -n 201
```
