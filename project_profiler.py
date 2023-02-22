import os

os.system("python enjoy.py --algo tqc --env TestEnvironment-v1  --env-kwargs render:True mode:"'random'" solve:False energy_command:0.3 -f logs/ -n 100
")