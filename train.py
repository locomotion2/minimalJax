import rl_zoo3
import rl_zoo3.train
from rl_zoo3.train import train
import src

from sbx import TQC, SAC

# Add new algorithm
rl_zoo3.ALGOS["tqc"] = TQC
rl_zoo3.ALGOS["sac"] = SAC
rl_zoo3.train.ALGOS = rl_zoo3.ALGOS
rl_zoo3.exp_manager.ALGOS = rl_zoo3.ALGOS

if __name__ == "__main__":
    train()
