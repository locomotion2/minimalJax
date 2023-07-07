import stable_baselines3.common.save_util as loader
from systems import dpendulum_utils

from hyperparams import settings

if __name__ == "__main__":
    print('Generating the dataset.')

    data = dpendulum_utils.generate_data(settings)

    print(f'Savig to {settings["data_dir"]}.')
    loader.save_to_pkl(path=settings["data_dir"], obj=data, verbose=1)

    print('Data generation finished.')
