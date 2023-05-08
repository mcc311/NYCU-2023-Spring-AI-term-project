from typing import Any
from env import ThreeByThreeGameEnv
from collections import defaultdict
import numpy as np
class NTupleAgent():
    def __init__(self) -> None:
        self.nets = [np.zeros((100,100,100))] * 5

    def extract_feature(self, state:np.ndarray):
        for i in range(3):
            yield state[:, i]
            yield state[i,:]
        yield state.diagonal()
        yield np.fliplr(state).diagonal()

    def __call__(self, state) -> Any:
        pass


if __name__ == "__main__":
    agent = NTupleAgent()
    env = ThreeByThreeGameEnv()
    obs, _ = env.reset()
    print(obs)
    for feat in agent.extract_feature(obs):
        print(feat)
