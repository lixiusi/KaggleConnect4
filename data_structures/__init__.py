from torch.utils.data.dataset import IterableDataset, Dataset


class ReplayMemory(IterableDataset):

    def __init__(self, agent):
        self.agent = agent

    def __getitem__(self, index):
        pass

    def __iter__(self):
        while True:
            yield self.agent.recall()


class Dummy(Dataset):

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return 0
