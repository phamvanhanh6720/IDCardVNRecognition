import os

import gdown
import yaml
from pydantic import BaseModel
try:
    from importlib import resources
except ImportError:
    import importlib_resources as resources


def download_weights(url, cache=None, md5=None, quiet=False):

    return os.path.join(gdown.cached_download(url, path=cache, md5=md5, quiet=quiet))


class Config(dict):
    def __init__(self, config):
        super(Config, self).__init__(**config)
        self.__dict__ = self

    @staticmethod
    def load_config():
        """
        Load config from config.yml file in face_recognition package
        Returns: Dict
        """
        with resources.open_text('app', 'config.yml') as ymlfile:
            cfg = yaml.safe_load(ymlfile)

        return Config(cfg)


class Item(BaseModel):
    base64_img: str
    key: str

