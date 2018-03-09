import os
import configparser

class Config(object):

    def __new__(cls, file):
        config = configparser.ConfigParser()
        if not os.path.exists(file):
            raise IOError("File {} does not exist".format(file))

        config.read(file)
        return config
