"""Pseudocode description of the AlphaZero algorithm."""
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import namedtuple
from typing import List, Tuple
from agent import Actor, Critic

##########################
####### Helpers ##########

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AlphaZeroConfig:

    def __init__(self):
        # Self-Play
        self.num_actors = 1

        self.num_sampling_moves = 5
        self.max_moves = 300  # for chess and shogi, 722 for Go.
        self.num_simulations = 8

        # Root prior exploration noise.
        # for chess, 0.03 for Go and 0.15 for shogi.
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # Training
        self.training_steps = int(700e1)
        self.checkpoint_interval = int(1e3)
        self.window_size = int(1e6)
        self.batch_size = 4096
        self.num_episode = 100

        self.weight_decay = 1e-4
        # Schedule for chess and shogi, Go starts at 2e-2 immediately.
        # self.learning_rate_schedule = {
        #     0: 2e-1,
        #     100e3: 2e-2,
        #     300e3: 2e-3,
        #     500e3: 2e-4
        # }
        self.learning_rate = 2e-1
        self.lr_step_size = 200e3
        self.gamma = 1e-1
        self.discount_factor = 0.99


class Node:

    def __init__(self, prior: float):
        Action = int
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = dict[Action, Node]()

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


class Game:

    def __init__(self, history: List[tuple[np.ndarray, int]] = None,
                 init_board: np.ndarray[int, (3, 3)] = None):
        self.history = history or list[tuple[np.ndarray, int]]()
        self.child_visits = []
        self.num_actions = 18
        if init_board is not None:
            self.board = init_board
        else:
            self.board = np.random.randint(0, 100, (3, 3))
        self.init_board = self.board.copy()

        self.bonus = 10
        self.penalty = -10

        self.boards = []

    def terminal(self):
        # Game specific termination rules.
        if ((~self.board.any(axis=0)).any() or  # contain any all 0s col
            (~self.board.any(axis=1)).any() or  # contain any all 0s row
            ~self.board.diagonal().any() or  # if diag is all 0s
                ~np.fliplr(self.board).diagonal().any()):  # if flipped diag is all 0s
            return True

        # 2nd termination condition
        elif (self.board == 0).any(axis=0).all():
            return True

        return False

    def terminal_value(self, to_play):
        # Game specific value.
        value = 0
        for i in range(to_play, len(self.history) - 1 + to_play, 2):
            _, action = self.history[i]
            value -= action % 3

        if len(self.history) % 2 != to_play:  # check if to_play is the last player
            value += self.terminal_reward()

        return value

    def terminal_reward(self):
        # 1st termination condition
        if ((~self.board.any(axis=0)).any() or  # contain any all 0s col
            (~self.board.any(axis=1)).any() or  # contain any all 0s row
            ~self.board.diagonal().any() or  # if diag is all 0s
                ~np.fliplr(self.board).diagonal().any()):  # if flipped diag is all 0s
            return self.bonus
        # 2nd termination condition
        elif (self.board == 0).any(axis=0).all():
            return self.penalty

        return 0

    def legal_actions(self):
        # Game specific calculation of legal actions.
        legal_actions = [action for action in range(self.num_actions)
                         if self.is_legal_move(action)]
        return legal_actions

    def is_legal_move(self, action):
        row_or_col, line, num_to_subtract = action//9, action % 9//3, action % 3
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

    def clone(self):
        return Game(list(self.history))

    def apply(self, action):
        row_or_col, line, num_to_subtract = action//9, action % 9//3, action % 3
        num_to_subtract += 1
        if row_or_col == 0:
            idx = np.s_[line, :]
        else:  # if row_or_col == 1
            idx = np.s_[:, line]
        self.board[idx] -= num_to_subtract
        self.history.append((self.board, action))

    def who_win(self):
        return np.argmax([self.terminal_value(0), self.terminal_value(1)])

    def store_search_statistics(self, root: Node):
        sum_visits = sum(
            child.visit_count for child in root.children.values())
        self.child_visits.append([
            root.children[a].visit_count /
            sum_visits if a in root.children else 0
            for a in range(self.num_actions)
        ])

    def make_image(self, state_index: int):
        # Game specific feature planes.
        if (state_index == -1):
            state = self.board
        else:
            state, _ = self.history[state_index]
        return torch.from_numpy(
            state[np.newaxis, :, :]).to(dtype=torch.float, device=device)

    def make_target(self, state_index: int):
        return (self.terminal_value(state_index % 2),
                self.child_visits[state_index])

    def to_play(self):
        return len(self.history) % 2


class ReplayBuffer:

    def __init__(self, config: AlphaZeroConfig):
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = list[Game]()

    def save_game(self, game):
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self.buffer.append(game)

    def sample_batch(self) -> List[Tuple[np.ndarray[int, (3,3)], Tuple[float, float]]] :
        # Sample uniformly across positions.
        move_sum = float(sum(len(g.history) for g in self.buffer))
        games = np.random.choice(
            self.buffer,
            size=self.batch_size,
            p=[len(g.history) / move_sum for g in self.buffer])
        game_pos = [(g, np.random.randint(len(g.history))) for g in games]
        return [(g.make_image(i), g.make_target(i)) for (g, i) in game_pos]


class Network(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.critic = Critic()
        self.actor = Actor()

    def forward(self, image):
        return (self.critic(image), self.actor(image))  # Value, Policy


class SharedStorage:

    def __init__(self):
        self._networks = {}

    def latest_network(self) -> Network:
        if self._networks:
            return self._networks[max(self._networks.keys())]
        else:
            return Network()

    def save_network(self, step: int, network: Network):
        self._networks[step] = network


##### End Helpers ########
##########################


# AlphaZero training is split into two independent parts: Network training and
# self-play data generation.
# These two parts only communicate by transferring the latest network checkpoint
# from the training to the self-play, and the finished games from the self-play
# to the training.
def alphazero(config: AlphaZeroConfig):
    storage = SharedStorage()
    replay_buffer = ReplayBuffer(config)

    for _ in range(config.num_actors):
        launch_job(run_selfplay, config, storage, replay_buffer)

    train_network(config, storage, replay_buffer)

    return storage.latest_network()


##################################
####### Part 1: Self-Play ########


# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.
def run_selfplay(config: AlphaZeroConfig, storage: SharedStorage,
                 replay_buffer: ReplayBuffer):
    for _ in range(config.num_episode):
        network = storage.latest_network()
        network.to(device)
        network.train()
        game = play_game(config, network)
        replay_buffer.save_game(game)


# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
def play_game(config: AlphaZeroConfig, network: Network):
    game = Game()
    while not game.terminal() and len(game.history) < config.max_moves:
        action, root = run_mcts(config, game, network)
        game.apply(action)
        game.store_search_statistics(root)
    return game


# Core Monte Carlo Tree Search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.
def run_mcts(config: AlphaZeroConfig, game: Game, network: Network):
    root = Node(0)
    evaluate(root, game, network)
    add_exploration_noise(config, root)

    for _ in range(config.num_simulations):
        node = root
        scratch_game = game.clone()
        search_path = [node]

        while node.expanded():
            action, node = select_child(config, node)
            scratch_game.apply(action)
            search_path.append(node)

        value = evaluate(node, scratch_game, network)
        backpropagate(search_path, value, scratch_game.to_play())
    return select_action(config, game, root), root


def select_action(config: AlphaZeroConfig, game: Game, root: Node):
    visit_counts = [(child.visit_count, action)
                    for action, child in root.children.items()]
    if len(game.history) < config.num_sampling_moves:
        logit = torch.Tensor(
            [visit_count for visit_count, _ in visit_counts]).to(dtype=float)
        probs = F.softmax(logit, dim=-1)
        dist = Categorical(probs)
        _, action = visit_counts[dist.sample()]

    else:
        _, action = max(visit_counts)
    return action


# Select the child with the highest UCB score.
def select_child(config: AlphaZeroConfig, node: Node):
    _, action, child = max((ucb_score(config, node, child), action, child)
                           for action, child in node.children.items())
    return action, child


# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(config: AlphaZeroConfig, parent: Node, child: Node):
    pb_c = math.log((parent.visit_count + config.pb_c_base + 1) /
                    config.pb_c_base) + config.pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior
    value_score = child.value()
    return prior_score + value_score


# We use the neural network to obtain a value and policy prediction.
def evaluate(node: Node, game: Game, network: Network):
    value, policy_logits = network(game.make_image(-1))
    value = value.squeeze(0)
    policy_logits = policy_logits.squeeze(0)

    # Expand the node.
    node.to_play = game.to_play()
    policy = {a: math.exp(policy_logits[a]) for a in game.legal_actions()}
    policy_sum = sum(policy.values())
    for action, p in policy.items():
        node.children[action] = Node(p / policy_sum)
    return value


# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(search_path: List[Node], value: float, to_play):
    for node in search_path:
        node.value_sum += value if node.to_play == to_play else (1 - value)
        node.visit_count += 1


# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(config: AlphaZeroConfig, node: Node):
    actions = node.children.keys()
    noise = np.random.gamma(config.root_dirichlet_alpha, 1, len(actions))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


######### End Self-Play ##########
##################################

##################################
####### Part 2: Training #########


def train_network(config: AlphaZeroConfig, storage: SharedStorage,
                  replay_buffer: ReplayBuffer):
    network = Network().to(device)
    optimizer = torch.optim.AdamW(network.parameters(),
                                  weight_decay=config.weight_decay,
                                  lr=config.learning_rate)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=config.lr_step_size,
                                                   gamma=config.gamma)
    network.train()
    for i in range(config.training_steps):
        print(i)
        if i % config.checkpoint_interval == 0:
            storage.save_network(i, network)
        batch = replay_buffer.sample_batch()
        update_weights(optimizer, network, batch, lr_scheduler)
    storage.save_network(config.training_steps, network)


def update_weights(optimizer: torch.optim.Optimizer, network: Network, 
                   batch: List[Tuple[torch.FloatTensor, Tuple[float, float]]],
                   lr_scheduler: torch.optim.lr_scheduler.LRScheduler):
    optimizer.zero_grad()
    states, targets = map(list, zip(*batch))
    states: List[torch.FloatTensor]
    target_value, target_policy = map(list, zip(*targets))
    values, policy_logits = network(torch.stack(states))
    loss = (
        F.smooth_l1_loss(values, torch.tensor(target_value).to(device).unsqueeze(1)) +
        F.cross_entropy(policy_logits, torch.tensor(
                target_policy).to(device))
    )

    # for image, (target_value, target_policy) in batch:
    #     value, policy_logits = network(image)
    #     loss += (
    #         F.smooth_l1_loss(value, torch.tensor(target_value).unsqueeze(0).to(device)) +
    #         F.cross_entropy(policy_logits, torch.tensor(
    #             target_policy).to(device))
    #     )
    loss.backward()
    optimizer.step()
    lr_scheduler.step()


######### End Training ###########
##################################

################################################################################
############################# End of pseudocode ################################
################################################################################


# Stubs to make the typechecker happy, should not be included in pseudocode
# for the paper.

def launch_job(f, *args):
    f(*args)


if __name__ == "__main__":
    config = AlphaZeroConfig()
    alphazero(config)
