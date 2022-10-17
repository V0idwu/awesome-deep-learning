#!/usr/bin python3
# -*- coding: utf-8 -*-
"""
@Time    :   2022/04/07 18:25:54
@Author  :   Tianyi Wu 
@Contact :   wutianyitower@hotmail.com
@File    :   alphazero.ipynb
@Version :   1.0
@Desc    :   None
"""

# here put the import lib
import collections
import logging
import math
import sys
from pickletools import pyint

import numpy as np

np.random.seed(0)
import gym
import pandas as pd
import torch

torch.manual_seed(0)
import boardgame2
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from boardgame2 import BLACK, WHITE
from gym.envs.registration import register

register(id="TicTacToe-v0", entry_point="boardgame2:BoardGameEnv")

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout, datefmt="%H:%M:%S"
)

print("torch.version:  ", torch.__version__)
print("device count:   ", torch.cuda.device_count())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("current device: ", device)
    print("device name:    ", torch.cuda.get_device_name(device))


env = gym.make(id="TicTacToe-v0", board_shape=3)
env.seed(0)
for key in vars(env):
    logging.info("%s: %s", key, vars(env)[key])
for key in vars(env.spec):
    logging.info("%s: %s", key, vars(env.spec)[key])


class AlphaZeroReplayer:
    def __init__(self):
        self.fields = ["player", "board", "prob", "winner"]
        self.memory = pd.DataFrame(columns=self.fields)

    def store(self, df):
        self.memory = pd.concat([self.memory, df[self.fields]], ignore_index=True)

    def sample(self, size):
        indices = np.random.choice(self.memory.shape[0], size=size)
        return (np.stack(self.memory.loc[indices, field]) for field in self.fields)


class AlphaZeroNet(nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        self.input_shape = input_shape

        # common net
        self.input_net = nn.Sequential(nn.Conv2d(1, 256, kernel_size=3, padding="same"), nn.BatchNorm2d(256), nn.ReLU())
        self.residual_nets = [
            nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding="same"), nn.BatchNorm2d(256)) for _ in range(2)
        ]

        # probability net
        self.prob_net = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding="same"),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=3, padding="same"),
        )

        # value net
        self.value_net0 = nn.Sequential(nn.Conv2d(256, 1, kernel_size=3, padding="same"), nn.BatchNorm2d(1), nn.ReLU())
        self.value_net1 = nn.Sequential(nn.Linear(np.prod(input_shape), 1), nn.Tanh())

    def forward(self, board_tensor):
        # common net
        input_tensor = board_tensor.view(-1, 1, *self.input_shape)
        x = self.input_net(input_tensor)
        for i_net, residual_net in enumerate(self.residual_nets):
            y = residual_net(x)
            if i_net == len(self.residual_nets) - 1:
                y = y + x
            x = torch.clamp(y, 0)
        common_feature_tensor = x

        # probability net
        logit_tensor = self.prob_net(common_feature_tensor)
        logit_flatten_tensor = logit_tensor.view(-1)
        prob_flatten_tensor = F.softmax(logit_flatten_tensor, dim=-1)
        prob_tensor = prob_flatten_tensor.view(-1, *self.input_shape)

        # value net
        v_feature_tensor = self.value_net0(common_feature_tensor)
        v_flatten_tensor = v_feature_tensor.view(-1, np.prod(self.input_shape))
        v_tensor = self.value_net1(v_flatten_tensor)

        return prob_tensor, v_tensor


class AlphaZeroAgent:
    def __init__(self, env):
        self.env = env
        self.board = np.zeros_like(env.board)
        self.reset_mcts()

        self.replayer = AlphaZeroReplayer()

        self.net = AlphaZeroNet(input_shape=self.board.shape)
        self.prob_loss = nn.BCELoss()
        self.v_loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters(), 1e-3, weight_decay=1e-4)

    def reset_mcts(self):
        def zero_board_factory():  # for construct default_dict
            return np.zeros_like(self.board, dtype=float)

        self.q = collections.defaultdict(zero_board_factory)
        # q estimates: board -> board
        self.count = collections.defaultdict(zero_board_factory)
        # q count visitation: board -> board
        self.policy = {}  # policy: board -> board
        self.valid = {}  # valid position: board -> board
        self.winner = {}  # winner: board -> None or int

    def reset(self, mode):
        self.mode = mode
        if mode == "train":
            self.trajectory = []

    def step(self, observation, winner, _):
        board, player = observation
        canonical_board = player * board
        s = boardgame2.strfboard(canonical_board)
        while self.count[s].sum() < 200:  # conduct MCTS 200 times
            self.search(canonical_board, prior_noise=True)
        prob = self.count[s] / self.count[s].sum()

        # sample
        location_index = np.random.choice(prob.size, p=prob.reshape(-1))
        action = np.unravel_index(location_index, prob.shape)

        if self.mode == "train":
            self.trajectory += [player, board, prob, winner]
        return action

    def close(self):
        if self.mode == "train":
            self.save_trajectory_to_replayer()
            if len(self.replayer.memory) >= 1000:
                for batch in range(2):  # learn multiple times
                    self.learn()
                self.replayer = AlphaZeroReplayer()  # reset replayer after the agent changes itself
                self.reset_mcts()

    def save_trajectory_to_replayer(self):
        df = pd.DataFrame(
            np.array(self.trajectory, dtype=object).reshape(-1, 4), columns=["player", "board", "prob", "winner"], dtype=object
        )
        winner = self.trajectory[-1]
        df["winner"] = winner
        self.replayer.store(df)

    def search(self, board, prior_noise=False):  # MCTS
        s = boardgame2.strfboard(board)

        if s not in self.winner:
            self.winner[s] = self.env.get_winner((board, BLACK))
        if self.winner[s] is not None:  # if there is a winner
            return self.winner[s]

        if s not in self.policy:  # leaf that has not calculate the policy
            board_tensor = torch.as_tensor(board, dtype=torch.float).view(1, 1, *self.board.shape)
            pi_tensor, v_tensor = self.net(board_tensor)
            pi = pi_tensor.detach().numpy()[0]
            v = v_tensor.detach().numpy()[0]
            valid = self.env.get_valid((board, BLACK))
            masked_pi = pi * valid
            total_masked_pi = np.sum(masked_pi)
            if total_masked_pi <= 0:
                # all valid actions do not have probabilities. rarely occur
                masked_pi = valid  # workaround
                total_masked_pi = np.sum(masked_pi)
            self.policy[s] = masked_pi / total_masked_pi
            self.valid[s] = valid
            return v

        # calculate PUCT
        count_sum = self.count[s].sum()
        c_init = 1.25
        c_base = 19652.0

        # λ(s,a)  = coef / count_sum = c(s) / self.count[s] = c(s,a)
        coef = (c_init + np.log1p((1 + count_sum) / c_base)) * math.sqrt(count_sum) / (1.0 + self.count[s])
        if prior_noise:
            alpha = 1.0 / self.valid[s].sum()
            noise = np.random.gamma(alpha, 1.0, board.shape)
            noise *= self.valid[s]
            noise /= noise.sum()
            prior_exploration_fraction = 0.25
            prior = (1.0 - prior_exploration_fraction) * self.policy[s] + prior_exploration_fraction * noise
        else:
            prior = self.policy[s]
        ub = np.where(self.valid[s], self.q[s] + coef * prior, np.nan)
        location_index = np.nanargmax(ub)
        location = np.unravel_index(location_index, board.shape)

        (next_board, next_player), _, _, _ = self.env.next_step((board, BLACK), np.array(location))
        next_canonical_board = next_player * next_board
        next_v = self.search(next_canonical_board)  # recursive
        v = next_player * next_v

        # s represents state of the board
        # location represents action
        self.count[s][location] += 1

        # q(s,a) <- q(s,a) + 1/c(s,a) * (G - q(s,a))
        self.q[s][location] += (v - self.q[s][location]) / self.count[s][location]
        return v

    def learn(self):
        players, boards, probs, winners = self.replayer.sample(64)
        canonical_boards = players[:, np.newaxis, np.newaxis] * boards  # experiences transform to one-player view
        targets = (players * winners)[:, np.newaxis]

        target_prob_tensor = torch.as_tensor(probs, dtype=torch.float)
        canonical_board_tensor = torch.as_tensor(canonical_boards, dtype=torch.float)
        target_tensor = torch.as_tensor(targets, dtype=torch.float)

        prob_tensor, v_tensor = self.net(canonical_board_tensor)

        flatten_target_prob_tensor = target_prob_tensor.view(-1, self.board.size)
        flatten_prob_tensor = prob_tensor.view(-1, self.board.size)
        prob_loss_tensor = self.prob_loss(flatten_prob_tensor, flatten_target_prob_tensor)
        v_loss_tensor = self.v_loss(v_tensor, target_tensor)
        loss_tensor = prob_loss_tensor + v_loss_tensor
        self.optimizer.zero_grad()
        loss_tensor.backward()
        self.optimizer.step()


def play_boardgame2_episode(env, agent, mode=None, verbose=False):
    observation, winner, done = env.reset(), 0, False
    agent.reset(mode=mode)
    elapsed_steps = 0
    while True:
        if verbose:
            board, player = observation
            print(boardgame2.strfboard(board))
        action = agent.step(observation, winner, done)
        if verbose:
            logging.info("step %d: player %d, action %s", elapsed_steps, player, action)
        observation, winner, done, _ = env.step(action)
        if done:
            if verbose:
                board, _ = observation
                print(boardgame2.strfboard(board))
            break
        elapsed_steps += 1
    agent.close()
    return winner, elapsed_steps


if __name__ == "__main__":

    agent = AlphaZeroAgent(env=env)

    for episode in range(5000):
        winner, elapsed_steps = play_boardgame2_episode(env, agent, mode="train")
        logging.info("train episode %d: winner = %d, steps = %d", episode, winner, elapsed_steps)

        if len(agent.replayer.memory) == 0:  # just finish learning
            logging.info("test episode %d:", episode)
            winner, elapsed_steps = play_boardgame2_episode(env, agent, mode="test", verbose=True)
            logging.info("test episode %d: winner = %d, steps = %d", episode, winner, elapsed_steps)
