import json

import lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split


class myDataset(Dataset):
    """
    A custom dataset class to handle data points.
    """

    def __init__(
        self,
        path_to_data,  # X,y pairs
    ):
        # Your data
        # e.g. np.memmap(path_to_data, dtype=np.uint16, mode='r')
        self.data = None

    def __len__(self):
        # TODO: implement based on your logic
        pass

    def __getitem__(self, idx):
        # TODO: implement your logic based on how data is store
        X, y = self.data[idx]
        return X, y

    def __iter__(self):
        # TODO: implement based on yuor logic
        for i in self.data:
            yield self.data[i]


class myDataModule(pl.LightningDataModule):
    """
    A custom datamodule to get the data, prepare it and return batches of X,y.
    """

    def __init__(self, config_path):
        super().__init__()

        # Read configuration file
        with open(config_path, "r") as f:
            self.config = json.load(f)

        # Set attributes
        self.path_to_data = self.config["path_to_data"]
        self.batch_size = self.config["batch_size"]
        self.train_val_test_split = self.config["train_val_test_split"]

        # Default
        self.num_workers = 2

    def prepare_data(self):
        # Download data
        # Process once at the beginning of training
        # Called on only 1 GPU (even in multi-GPU scenarios)
        # Do not set state here (like self.x = y) as it won't be available across GPUs
        pass

    def setup(self, stage=None):
        # Prepare data for different stages (fit/test/predict)
        # Split data into train/val/test sets
        # Set internal state (this is where you can assign to self)
        # Called on every GPU separately

        # TODO: custom processing (e.g. masking ...)

        # Init a dataset
        self.data = myDataset(self.path_to_data)

        # Split the dataset
        self.train_ds, self.val_ds, self.test_ds = random_split(
            self.data, self.train_val_test_split
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )
