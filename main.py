import jsonargparse

from src.conf import MainConfig
from src.train import Trainer


class Main:
    def __init__(self, conf: MainConfig = MainConfig()):
        self.conf = conf

    def train(self):
        trainer = Trainer(self.conf)
        trainer.train()


if __name__ == "__main__":
    jsonargparse.CLI(Main)
