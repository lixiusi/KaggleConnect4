import os
import argparse

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import algorithms


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", default="DQN")

    args = parser.parse_args()

    model = getattr(algorithms, args.algorithm)()
    logger = TensorBoardLogger(
        save_dir=os.path.join(os.getcwd(), "logs_and_ckpts"),
        name=args.algorithm
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=logger.log_dir,
        every_n_train_steps=500
    )

    trainer = Trainer(
        accelerator="mps",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback],
        val_check_interval=500
    )

    trainer.fit(model)
