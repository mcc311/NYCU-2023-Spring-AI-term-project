from agent import Actor, Critic, ConvActor, ConvCritic
from env import ThreeByThreeGameEnv
import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
import torch.distributions as distributions
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

import numpy as np

LEARNING_RATE = 0.00005
MAX_EPISODES = 1_000
DISCOUNT_FACTOR = 0.99
TRACE_DECAY = 0.9
N_TRIALS = 25
REWARD_THRESHOLD = 200
PRINT_EVERY = 10
TRAIN_POLICY_EVERY = 100


def to_action(idx: int):
    return (idx//9, idx % 9//3, idx % 3)


def calculate_returns(rewards, discount_factor, normalize=True):
    returns = []
    R = 0

    for r in reversed(rewards):
        R = r + R * discount_factor
        returns.insert(0, R)

    returns = torch.tensor(returns)

    if normalize:

        returns = (returns - returns.mean()) / returns.std()

    return returns


def calculate_advantages(rewards, values, discount_factor, trace_decay, normalize=True):
    advantages = []
    advantage = 0
    next_value = 0

    for r, v in zip(reversed(rewards), reversed(values)):
        td_error = r + next_value * discount_factor - v
        advantage = td_error + advantage * discount_factor * trace_decay
        next_value = v
        advantages.insert(0, advantage)

    advantages = torch.tensor(advantages)

    if normalize:
        advantages = (advantages - advantages.mean()) / advantages.std()

    return advantages


def test_an_episode(env: ThreeByThreeGameEnv, 
                    actor: nn.Module, critic: nn.Module):
    actor.eval()
    critic.eval()

    done = False
    ep_reward = 0

    observation, _ = env.reset()
    init_state = observation.copy()
    expected_min_cost = np.min([init_state.min(axis=0).sum(), init_state.min(axis=1).sum()])
    while not done:

        observation = torch.from_numpy(
            observation[np.newaxis,:,:]).to(dtype=torch.float)
        action_logit = actor(observation)
        action_prob = F.softmax(action_logit, dim=-1)
        for action_index in action_prob.argsort(descending=True):
            action = to_action(action_index.detach().item())
            if env.is_legal_move(action):
                break

        observation, reward, done, info = env.step(action)
        ep_reward += reward


    print(f"test reward:{ep_reward} | ratio: {ep_reward/expected_min_cost} | bonus: {info['t1']}")


def play_an_episode(env: ThreeByThreeGameEnv, 
                    actor: nn.Module, critic: nn.Module,
                    actor_optim: optim.Optimizer, critic_optim: optim.Optimizer, iter: int):
    actor.train()
    critic.train()

    log_prob_actions = []
    values = []
    rewards = []
    done = False
    ep_reward = 0

    observation, _ = env.reset()
    init_state = observation.copy()
    expected_min_cost = np.min([init_state.min(axis=0).sum(), init_state.min(axis=1).sum()])
    actor_optim.zero_grad()
    critic_optim.zero_grad()
    while not done:

        observation = torch.from_numpy(
            observation[np.newaxis,:,:]).to(dtype=torch.float)
        value = critic(observation)
        action_logit = actor(observation)
        action_prob = F.softmax(action_logit, dim=-1)
        dist = distributions.Categorical(action_prob)

        gumbel_prob = F.gumbel_softmax(action_prob.log(), tau=1, dim=-1, hard=False)
        for action_index in gumbel_prob.argsort(descending=True):
            action = to_action(action_index.detach().item())
            if env.is_legal_move(action):
                break

        log_prob_action = dist.log_prob(action_index)
        observation, reward, done, _ = env.step(action)
        log_prob_actions.append(log_prob_action)
        values.append(value)
        rewards.append(reward)
        ep_reward += reward
    log_prob_actions = torch.stack(log_prob_actions)
    values = torch.cat(values).squeeze(-1)

    returns = calculate_returns(rewards, DISCOUNT_FACTOR)
    # note: calculate_advantages takes in rewards, not returns!
    advantages = calculate_advantages(
        rewards, values, DISCOUNT_FACTOR, TRACE_DECAY)


    advantages = advantages.detach()
    returns = returns.detach()


    policy_loss = - (advantages * log_prob_actions).sum()
    value_loss = F.smooth_l1_loss(returns, values).sum()


    if iter % TRAIN_POLICY_EVERY == 0:
        policy_loss.backward()
        actor_optim.step()

    value_loss.backward()
    critic_optim.step()

    return policy_loss.item(), value_loss.item(), \
    {'ep_reward': ep_reward, 'init_state': init_state, \
     'expected_min_cost': expected_min_cost}


if __name__ == "__main__":
    env = ThreeByThreeGameEnv()
    actor = ConvActor(env)
    critic = ConvCritic(env)
    actor_optimizer = optim.AdamW(actor.parameters(), lr=LEARNING_RATE)
    critic_optimizer = optim.AdamW(actor.parameters(), lr=LEARNING_RATE)
    plosses = []
    vlosses = []
    ewma = 0
    for i_episode in range(int(1e6)):
        ploss, vloss, info = play_an_episode(env, actor, critic, actor_optimizer, critic_optimizer, i_episode)
        ep_reward, expected_min_cost = info['ep_reward'], info['expected_min_cost']
        plosses.append(ploss)
        vlosses.append(vloss)
        ewma = 0.9*ewma + 0.1+ep_reward
        writer.add_scalar('Policy/Loss', np.mean(ploss), i_episode)
        writer.add_scalar('Critic/Loss', np.mean(vloss), i_episode)
        writer.add_scalar('EWMA', ewma, i_episode)
        writer.add_scalar('ratio', ep_reward/expected_min_cost, i_episode)
        if i_episode % PRINT_EVERY == 0 and i_episode != 0:
            print(f"{i_episode} th | P-Loss: {ploss:.2f} | V-Loss: {vloss:.2f} | Total Return: {ep_reward} | EMC: {expected_min_cost} | ratio: {ep_reward/expected_min_cost}")
            test_an_episode(env, actor, critic)
            plosses = []
            vlosses = []
