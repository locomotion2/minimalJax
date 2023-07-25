from hyperparams import settings
import identification.systems.dpendulum_utils as dutils
import sqlite3

if __name__ == '__main__':
    # setup database
    filepath = f"{settings['data_settings']['data_dir']}"
    database = sqlite3.connect(filepath)

    # setup new database and scramble
    filepath_target = f"{filepath}_scrambled"
    with open(filepath_target, 'w'):
        database_target = sqlite3.connect(filepath_target)
        dutils.scramble_data(database, database_target, settings)

    # close everything
    database_target.commit()
    database.close()
    database_target.close()
