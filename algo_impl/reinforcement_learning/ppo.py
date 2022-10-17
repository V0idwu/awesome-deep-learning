import sys
import logging
import itertools

import numpy as np

np.random.seed(0)
import pandas as pd
import gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.distributions as distributions
import scipy.signal as signal

torch.manual_seed(0)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    stream=sys.stdout,
    # filemode='w',
    # filename='log_{}.log'.format{time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))},
    level=logging.DEBUG,
)


env = gym.make("Acrobot-v1")
env.seed(0)
for key in vars(env):
    logging.info("%s: %s", key, vars(env)[key])
for key in vars(env.spec):
    logging.info("%s: %s", key, vars(env.spec)[key])

print(env.action_space)
print(env.observation_space)


class PPOReplayer:
    def __init__(self) -> None:
        self.fields = ["state", "action", "prob", "advantage", "return"]
        self.memory = pd.DataFrame(columns=self.fields)

    def store(self, df):
        self.memory = pd.concat([self.memory, df[self.fields]], ignore_index=True)

    def sample(self, size):
        indices = np.random.choice(self.memory.shape[0], size=size)
        return (np.stack(self.memory.loc[indices, field]) for field in self.fields)


class PPOAgent:
    def __init__(self, env):
        self.gamma = 0.99

        self.replayer = PPOReplayer()

        self.actor_net = self.build_net(
            input_size=env.observation_space.shape[0],
            hidden_sizes=[100,],
            output_size=env.action_space.n,
            output_activator=nn.Softmax(1),
        )
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 0.001)
        self.critic_net = self.build_net(input_size=env.observation_space.shape[0], hidden_sizes=[100,],)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), 0.002)
        self.critic_loss = nn.MSELoss()

    def build_net(self, input_size, hidden_sizes, output_size=1, output_activator=None):
        layers = []
        for input_size, output_size in zip([input_size,] + hidden_sizes, hidden_sizes + [output_size,],):
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ReLU())
        layers = layers[:-1]
        if output_activator:
            layers.append(output_activator)
        net = nn.Sequential(*layers)
        return net

    def reset(self, mode=None):
        self.mode = mode
        if self.mode == "train":
            self.trajectory = []

    def step(self, observation, reward, done):
        state_tensor = torch.as_tensor(observation, dtype=torch.float).unsqueeze(0)
        prob_tensor = self.actor_net(state_tensor)
        action_tensor = distributions.Categorical(prob_tensor).sample()
        action = action_tensor.numpy()[0]
        if self.mode == "train":
            self.trajectory += [observation, reward, done, action]
        return action

    def close(self):
        if self.mode == "train":
            self.save_trajectory_to_replayer()
            if len(self.replayer.memory) >= 1000:
                for batch in range(5):  # learn multiple times
                    self.learn()
                self.replayer = PPOReplayer()  # reset replayer after the agent changes itself

    def save_trajectory_to_replayer(self):
        df = pd.DataFrame(np.array(self.trajectory, dtype=object).reshape(-1, 4), columns=["state", "reward", "done", "action"])
        state_tensor = torch.as_tensor(np.stack(df["state"]), dtype=torch.float)  # xxx * 6
        action_tensor = torch.as_tensor(df["action"], dtype=torch.long)  # xxx
        v_tensor = self.critic_net(state_tensor)  # xxx * 1
        df["v"] = v_tensor.detach().numpy()
        prob_tensor = self.actor_net(state_tensor)  # xxx * 3
        pi_tensor = prob_tensor.gather(-1, action_tensor.unsqueeze(1)).squeeze(1)  # 选择后为 xxx
        df["prob"] = pi_tensor.detach().numpy()
        df["next_v"] = df["v"].shift(-1).fillna(0.0)
        df["u"] = df["reward"] + self.gamma * df["next_v"]
        df["delta"] = df["u"] - df["v"]  # advantage function
        df["advantage"] = signal.lfilter([1.0,], [1.0, -self.gamma], df["delta"][::-1],)[::-1]
        df["return"] = signal.lfilter([1.0,], [1.0, -self.gamma], df["reward"][::-1],)[::-1]
        self.replayer.store(df)

    def learn(self):
        states, actions, old_pis, advantages, returns = self.replayer.sample(size=64)
        state_tensor = torch.as_tensor(states, dtype=torch.float)  # 64 * 6
        action_tensor = torch.as_tensor(actions, dtype=torch.long)  # 64
        old_pi_tensor = torch.as_tensor(old_pis, dtype=torch.float)  # 64
        advantage_tensor = torch.as_tensor(advantages, dtype=torch.float)  # 64
        return_tensor = torch.as_tensor(returns, dtype=torch.float).unsqueeze(1)  # 64 * 1

        # train actor
        all_pi_tensor = self.actor_net(state_tensor)  # 64 * 3
        pi_tensor = all_pi_tensor.gather(1, action_tensor.unsqueeze(1)).squeeze(1)  # 64
        surrogate_advantage_tensor = (pi_tensor / old_pi_tensor) * advantage_tensor
        clip_times_advantage_tensor = 0.1 * surrogate_advantage_tensor
        max_surrogate_advantage_tensor = advantage_tensor + torch.where(
            advantage_tensor > 0.0, clip_times_advantage_tensor, -clip_times_advantage_tensor
        )  # ε|a|  取绝对值
        clipped_surrogate_advantage_tensor = torch.min(surrogate_advantage_tensor, max_surrogate_advantage_tensor)
        actor_loss_tensor = -clipped_surrogate_advantage_tensor.mean()
        self.actor_optimizer.zero_grad()
        actor_loss_tensor.backward()
        self.actor_optimizer.step()

        # train critic
        pred_tensor = self.critic_net(state_tensor)
        critic_loss_tensor = self.critic_loss(pred_tensor, return_tensor)
        self.critic_optimizer.zero_grad()
        critic_loss_tensor.backward()
        self.critic_optimizer.step()


agent = PPOAgent(env)


def play_episode(env, agent, max_episode_steps=None, mode=None, render=False):
    observation, reward, done = env.reset(), 0.0, False
    agent.reset(mode=mode)
    episode_reward, elapsed_steps = 0.0, 0
    while True:
        action = agent.step(observation, reward, done)
        if render:
            env.render()
        if done:
            break
        observation, reward, done, _ = env.step(action)
        episode_reward += reward
        elapsed_steps += 1
        if max_episode_steps and elapsed_steps >= max_episode_steps:
            break
    agent.close()
    return episode_reward, elapsed_steps


if __name__ == "__main__":

    logging.info("==== train ====")
    episode_rewards = []
    for episode in itertools.count():
        episode_reward, elapsed_steps = play_episode(
            env.unwrapped, agent, max_episode_steps=env._max_episode_steps, mode="train"
        )
        episode_rewards.append(episode_reward)
        logging.debug("train episode %d: reward = %.2f, steps = %d", episode, episode_reward, elapsed_steps)
        if np.mean(episode_rewards[-10:]) > -120:
            break
    plt.plot(episode_rewards)

    logging.info("==== test ====")
    episode_rewards = []
    for episode in range(100):
        episode_reward, elapsed_steps = play_episode(env, agent)
        episode_rewards.append(episode_reward)
        logging.debug("test episode %d: reward = %.2f, steps = %d", episode, episode_reward, elapsed_steps)
    logging.info("average episode reward = %.2f ± %.2f", np.mean(episode_rewards), np.std(episode_rewards))

