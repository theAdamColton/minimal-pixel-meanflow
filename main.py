import jsonargparse

from src.conf import MainConfig
from src.train import train


class Main:
    def __init__(self, conf: MainConfig = MainConfig()):
        self.conf = conf

    def train(self):
        train(self.conf)


if __name__ == "__main__":
    jsonargparse.CLI(Main)
