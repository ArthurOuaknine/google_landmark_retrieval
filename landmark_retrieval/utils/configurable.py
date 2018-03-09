from landmark_retrieval.utils.config import Config

class Configurable(object):
    def __init__(self, path_to_config):
        self.config = Config(path_to_config)
