from src import lagranx as lx
from hyperparams import settings
import stable_baselines3.common.save_util as loader

if __name__ == "__main__":
    print('Generating the dataset.')

    data = lx.generate_data(settings)

    print(f'Savig to {settings["data_dir"]}.')
    loader.save_to_pkl(path=settings["data_dir"], obj=data, verbose=1)

    print('Data generation finished.')
