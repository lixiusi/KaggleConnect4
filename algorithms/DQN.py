from copy import deepcopy

from einops import rearrange
import torch
from torch import nn
from torch.utils.data import DataLoader

from algorithms.base import RLModel
from models import ConvNet
from agents.DQN import DQAgent

__all__ = ['DQN', 'DoubleDQN', 'ExpectedSarsa']


class DQN(RLModel):
    q_net: nn.Module
    target_q_net: nn.Module
    agent: DQAgent

    def __init__(
            self,
            lr: float = 1e-5,
            batch_size: int = 32,
            exploration_rate: float = 0.85,
            exploration_rate_decay: float = 0.99,
            exploration_rate_min: float = 0.01,
            returns_discount_gamma: float = 0.75,
            training_opponent: str = "negamax",
            replay_memory_capacity: int = 200,
            recall_n_steps: int = 5,
            sync_target_network_every: int = 50,
            q_net_conv_out_channels: int = 16,
            q_net_mlp_hidden_size: int = 128,
            log_example_game_every_n_steps: int = 5000
    ):
        super(DQN, self).__init__()

    def build_neural_nets(self):
        """Initialise neural network architectures used to estimate Q-values"""
        self.q_net = ConvNet(
            n_row=self.env.configuration.rows,
            n_col=self.env.configuration.columns,
            conv_out_channels=self.hparams.q_net_conv_out_channels,
            hidden_size=self.hparams.q_net_mlp_hidden_size
        )
        self.target_q_net = deepcopy(self.q_net)
        for p in self.target_q_net.parameters():
            p.requires_grad = False

    def setup_agent(self):
        """Instantiate DQAgent and fill its memory up to capacity"""
        self.agent = DQAgent(
            q_net=self.q_net,
            memory_capacity=self.hparams.replay_memory_capacity,
            recall_n_steps=self.hparams.recall_n_steps
        )
        self.agent.init_memory(self.env.configuration, self.train_env)

    def train_dataloader(self):
        """Stream training data from agent memory"""
        return DataLoader(dataset=self.agent.memory_dataset, batch_size=self.hparams.batch_size)

    @torch.no_grad()
    def on_train_batch_start(self, *args, **kwargs):
        """Update exploration rate and target network, before making a move and adding the experience to memory"""
        self.hparams.exploration_rate *= self.hparams.exploration_rate_decay
        self.hparams.exploration_rate = max(self.hparams.exploration_rate, self.hparams.exploration_rate_min)
        self.logger.log_hyperparams({"exploration_rate": self.hparams.exploration_rate})

        if self.global_step % self.hparams.sync_target_network_every == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.q_net.eval()
        self.state = self.agent.play(self.state, self.env.configuration, self.train_env, self.hparams.exploration_rate)
        self.q_net.train()

    def training_step(self, experience_batch, batch_idx):
        """
        Sample a batch of previous experiences from memory and compute their respective TD errors wrt current online
        policy
        """
        _, states, actions, rewards, dones, next_states = experience_batch
        b, n = actions.shape

        # an experience should be truncated if episode ends before n-steps are complete
        dones = torch.cat((dones, torch.ones((b, 1), device=self.device)), dim=1)
        seq_lens = torch.argmax(dones, dim=1)
        valid_steps = torch.stack([torch.cat((torch.ones(l), torch.zeros(n - l))) for l in seq_lens]).to(self.device)
        truncation_mask = ~valid_steps.bool()
        rewards = rewards.masked_fill_(truncation_mask, 0)

        # TD target (i.e. an estimate of n-step expected return) is a weighted sum of rewards throughout the
        # experience, as well as Q-value estimates computed by the target network at the end of the experience,
        # if the episode has not yet terminated by then
        last_states_q_vals = self.target_q_estimate(next_states[:, -1]).masked_fill(seq_lens != n, 0)
        rewards_and_q_vals = torch.cat((rewards, last_states_q_vals.unsqueeze(1)), dim=1)
        discounts = torch.cumprod(torch.ones(n) * self.hparams.returns_discount_gamma, dim=0).to(self.device)
        td_targets = torch.stack(
            [rewards[:, 0] + (discounts[:n - i] * rewards_and_q_vals[:, i + 1:]).sum(dim=1) for i in range(n)], dim=1
        )

        # Q-value estimates are computed by the current online policy
        online_q_estimates = self.q_net(rearrange(states, 'b n s -> (b n) s'))
        online_q_estimates = online_q_estimates.gather(-1, rearrange(actions, 'b n -> (b n) ()'))
        online_q_estimates = rearrange(online_q_estimates, '(b  n) () -> b n', b=b)

        # MSE loss is computed between Q-value estimates and TD targets at valid time steps
        loss = nn.MSELoss()(
            online_q_estimates.masked_fill_(truncation_mask, 0),
            td_targets.masked_fill_(truncation_mask, 0)
        )
        self.log("TD_error", loss, prog_bar=True)
        return loss

    @torch.no_grad()
    def target_q_estimate(self, states):
        return self.target_q_net(states).max(1)[0]


class DoubleDQN(DQN):

    @torch.no_grad()
    def target_q_estimate(self, states):
        q_all_actions = self.target_q_net(states)
        greedy_actions = self.agent.greedy_actions(states)
        return q_all_actions.gather(1, greedy_actions.unsqueeze(1)).squeeze()


class ExpectedSarsa(DQN):

    @torch.no_grad()
    def target_q_estimate(self, states):
        q_all_actions = self.target_q_net(states)
        greedy_actions = torch.max(q_all_actions, dim=1)[0]
        eps, b, na = self.hparams.exploration_rate, states.shape[0], self.n_cols
        p_explore = (eps / na) * torch.ones((b, na), device=self.device)
        p_exploit = p_explore + (1 - eps) * torch.ones((b, na), device=self.device)
        eps_greedy_policy = torch.where(q_all_actions == greedy_actions.unsqueeze(1), p_exploit, p_explore)
        expected_q = torch.einsum("ba, ba -> b", eps_greedy_policy, q_all_actions)
        return expected_q
