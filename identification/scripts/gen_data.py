import sqlite3

import stable_baselines3.common.save_util as loader
from identification.systems import dpendulum_utils

from identification.hyperparams import settings

if __name__ == "__main__":
    # setup new database
    filepath = settings['data_settings']['data_dir']
    with open(filepath, 'w'):
        database = sqlite3.connect(f"{filepath}")

        print("Generating the dataset.")
        dataframe = dpendulum_utils.generate_random_data(settings)
        dataframe.to_sql("scrambled_data", database, if_exists='replace', index=False)

    # print(f'Savig to {settings["data_settings"]["data_dir"]}.')
    # loader.save_to_pkl(path=settings["data_settings"]["data_dir"], obj=data, verbose=1)

    # close everything
    database.close()
    print("Data generation finished.")
