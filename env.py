import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ThreeByThreeGameEnv(gym.Env):
    def __init__(self):
        super(gym.Env, self).__init__()
        self.board = None
        self.player = None
        self.reward_range = (float('-inf'), float('inf'))
        self.action_space = spaces.Tuple((spaces.Discrete(2),  # row=0 or col=1
                                          spaces.Discrete(3),  # which line
                                          spaces.Discrete(3)))  # number to substract
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(3, 3), dtype=int)
        self.BONUS = 20
        self.PENALTY = 10

    def reset(self):
        # TODO: check for the rules(TBD)
        self.board = np.random.randint(1, 100, size=(3, 3))
        self.player = 0
        return self.board, {}
    
    def is_legal_move(self, action):
        row_or_col, line, num_to_subtract = action
        num_to_subtract += 1

        if (row_or_col not in range(2) or
                line not in range(3) or
                num_to_subtract not in range(1, 4)):
            return False
            # raise ValueError("Invalid action")

        if row_or_col == 0:
            idx = np.s_[line, :]
        else:  # if row_or_col == 1
            idx = np.s_[:, line]

        if 0 in self.board[idx]:
            return False
            # raise ValueError("Chosen row or col contains 0")

        if self.board[idx].min() < num_to_subtract:
            return False
            # raise ValueError(
            #     f"Cannot subtract {num_to_subtract} from target {self.board[idx]}")
        
        return True

    def step(self, action):
        row_or_col, line, num_to_subtract = action
        num_to_subtract += 1

        if (row_or_col not in range(2) or
                line not in range(3) or
                num_to_subtract not in range(1, 4)):
            raise ValueError("Invalid action")

        if row_or_col == 0:
            idx = np.s_[line, :]
        else:  # if row_or_col == 1
            idx = np.s_[:, line]

        if 0 in self.board[idx]:
            raise ValueError("Chosen row or col contains 0")

        if self.board[idx].min() < num_to_subtract:
            raise ValueError(
                f"Cannot subtract {num_to_subtract} from target {self.board[idx]}")

        reward = -num_to_subtract
        self.board[idx] -= num_to_subtract
        done = False

        term1 = False
        term2 = False
        # 1st termination condition
        if ((~self.board.any(axis=0)).any() or  # contain any all 0s col
            (~self.board.any(axis=1)).any() or  # contain any all 0s row
            ~self.board.diagonal().any() or  # if diag is all 0s
                ~np.fliplr(self.board).diagonal().any()):  # if flipped diag is all 0s
            reward += self.BONUS
            done = True
            term1 = True

        # 2nd termination condition
        if not done and (self.board == 0).any(axis=0).all():
            done = True
            reward -= self.PENALTY
            term2 = True

        return self.board, reward, done, {'t1':term1, 't2':term2}

    def render(self, mode='human'):
        print(self.board)

    def seed(seed):
        np.random.seed(seed)


if __name__ == '__main__':

    env = ThreeByThreeGameEnv()
    observation, info = env.reset()
    for _ in range(1000):
        env.render()
        action = env.action_space.sample()
        print(action)
        observation, reward, terminated, info = env.step(action)
        if terminated:
            observation, info = env.reset()
    env.close()
