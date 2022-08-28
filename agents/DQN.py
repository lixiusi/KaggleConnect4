from collections import deque, namedtuple
from tqdm import tqdm
import numpy as np
import torch

from data_structures import ReplayMemory
from utils import action_space

Experience = namedtuple("Experience", field_names=["time", "state", "action", "reward", "done", "new_state"])


@torch.no_grad()
class DQAgent:

    def __init__(self, q_net, memory_capacity, recall_n_steps=1):
        self.q_net = q_net
        self.n_row, self.n_col = q_net.n_row, q_net.n_col
        self.memory_capacity = memory_capacity
        self._memory = deque(maxlen=(memory_capacity + recall_n_steps))
        self.memory_dataset = ReplayMemory(self)
        self.recall_n_steps = recall_n_steps
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.time = 0
        self.every_recall = {}

    def init_memory(self, env_config, train_env):
        """Initialises replay memory"""
        state = train_env.reset()
        print(f'\nFilling experience replay memory buffer')
        for _ in tqdm(range(self.memory_capacity + self.recall_n_steps)):
            state = self.play(state, env_config, train_env, eps=1)

    def act(self, observation, env_config, eps: float = 0.0):
        """Given a state, choose an epsilon-greedy action"""
        if np.random.random() < eps:
            action = int(np.random.choice(action_space(observation, env_config)))
        else:
            state = torch.tensor([observation.board]).to(self.device)
            action = int(self.greedy_actions(state)[0])
        assert action in action_space(observation, env_config)
        return action

    def greedy_actions(self, states):
        """Find greedy actions given batch of states as a tensor"""
        boards = states.clone().detach().view(-1, self.n_row, self.n_col)
        q_values = self.q_net(states)
        valid_q_values = q_values.masked_fill_(~torch.any(boards == 0, dim=1), -1)
        actions = torch.argmax(valid_q_values, dim=1)
        return actions

    def cache(self, experience):
        """Add the experience to memory"""
        self._memory.append(experience)

    def recall(self):
        """Sample experiences from memory"""
        idx = np.random.choice(self.memory_capacity)
        times, states, actions, rewards, dones, next_states = zip(*[self._memory[idx + j] for j in range(self.recall_n_steps)])
        assert all(r in [-1, 0, 1] for r in rewards), f"Found something wrong with rewards in recall:\n{rewards}"

        experience_batch = (
            torch.tensor(times, device=self.device),
            torch.tensor(states, device=self.device),
            torch.tensor(actions, device=self.device),
            torch.tensor(rewards, dtype=torch.float32, device=self.device),
            torch.tensor(dones, dtype=torch.long, device=self.device),
            torch.tensor(next_states, device=self.device)
        )
        return experience_batch

    def play(self, state, env_config, train_env, eps: float = 0.0):
        """Interacts with environment, and cache the subsequent experience"""
        self.time += 1
        action = self.act(state, env_config, eps)
        new_state, reward, done, _ = train_env.step(action)
        experience = Experience(self.time, state.board, action, reward, done, new_state.board)
        self.cache(experience)
        return train_env.reset() if done else new_state
