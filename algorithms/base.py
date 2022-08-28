import kaggle_environments
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
from utils import win_percentage, visualize_board_and_q_values
from data_structures import Dummy


class RLModel(LightningModule):

    def __init__(self, *args, **kwargs):
        super(RLModel, self).__init__()
        self.save_hyperparameters()

        self.env = kaggle_environments.make("connectx")
        self.n_rows, self.n_cols = self.env.configuration.rows, self.env.configuration.columns
        self.train_env = self.env.train([None, self.hparams.training_opponent])
        self.state = self.train_env.reset()

        self.build_neural_nets()
        self.setup_agent()

    def val_dataloader(self):
        return DataLoader(dataset=Dummy(), batch_size=self.hparams.batch_size)

    def configure_optimizers(self):
        optimizer = Adam(self.q_net.parameters(), lr=self.hparams.lr)
        return optimizer

    def validation_step(self, *args, **kwargs):
        self.logger.log_metrics({
            f"p_win_vs_{opp}": win_percentage(self.agent.act, opp, num_episodes=10)
            for opp in ["random", "negamax"]
        })
        self.log_model_weight_histograms()
        if self.global_step % self.hparams.log_example_game_every_n_steps == 0:
            for opp in ["random", "negamax"]:
                self.log_example_game(opp)

    def log_model_weight_histograms(self):
        for name, weights in self.target_q_net.named_parameters():
            self.logger.experiment.add_histogram(
                f'model_weights/{name}', weights.clone().detach().cpu(), global_step=self.global_step)

    def log_example_game(self, opponent="negamax"):
        env = kaggle_environments.make("connectx")
        train_env = env.train([None, opponent])
        state, done = train_env.reset(), False
        time = 0
        while not done:
            board = torch.tensor([state.board], device=self.device)
            q_values = self.q_net(board)
            action = self.agent.act(state, env.configuration)
            state, reward, done, _ = train_env.step(action)
            fig = visualize_board_and_q_values(board, q_values.squeeze())
            self.logger.experiment.add_figure(
                f"example_game_vs_{opponent}/step={self.global_step}", fig, global_step=time
            )
            time += 1
