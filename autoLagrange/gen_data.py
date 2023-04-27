import lagranx as lx
import numpy as np
import stable_baselines3.common.save_util as loader

if __name__ == "__main__":
    print('Generating the dataset.')

    # Define all settings
    settings = {'batch_size': 100,
                'test_every': 10,
                'num_batches': 1500,
                'sections_num': 10,
                'time_step': 0.05,
                'data_size': 1000 * 100,
                'starting_point': np.array([3 * np.pi / 7, 3 * np.pi / 4, 0, 0], dtype=np.float32),
                'data_dir': 'tmp/data',
                'seed': 0
                }

    # Generate data
    data = lx.generate_data(settings)

    print(f'Savig to {settings["data_dir"]}.')
    loader.save_to_pkl(path=settings["data_dir"], obj=data, verbose=1)

    print('Data generation finished.')
