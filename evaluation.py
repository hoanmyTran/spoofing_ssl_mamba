import pytorch_lightning as pl
import torch
from typing import Optional
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import librosa
import numpy as np
import pandas as pd

def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

def generate_filelist(metadata_file):
    """
    Generates a list of spoofed audio file names from a metadata file.

    Args:
        dir_meta (str): Path to the metadata file. The file should contain lines
                        where each line represents a record with fields separated by spaces.
                        The second field in each line is assumed to be the file name.

    Returns:
        list: A list of audio file names.
    """

    df = pd.read_csv(metadata_file, usecols=['file_name','label'])

    return df.file_name.values


class DatasetEval(Dataset):
    """
    Dataset class for evaluation dataset.

    This class is responsible for loading and transforming the audio files
    for the evaluation dataset.

    Args:
        list_IDs (list of str): List of utterance IDs.
        base_dir (str): Path to the directory containing audio files.
        transform (callable, optional): A function/transform to apply to the data.
    """
    def __init__(self, file_paths, transform=None):
        """self.file_paths	: list of strings (each string: utt key),
            self.labels      : dictionary (key: utt key, value: label integer)"""
        self.file_paths = file_paths
        self.transform = transform
        if self.transform:
            self.transform = transforms.Compose([
                lambda x: pad(x),
                lambda x: torch.Tensor(x),
            ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        """
        Retrieves an audio file and its corresponding utterance ID.

        Args:
            index (int): Index of the audio file to retrieve.

        Returns:
            tuple: A tuple containing:
                - X (Tensor): The audio data.
                - utt_id (str): The utterance ID.
        """
        file_path = self.file_paths[index]
        X, fs = librosa.load(file_path, sr=16000)
        if self.transform:
            X = self.transform(X)
        else:
            X = torch.Tensor(X)

        return X, file_path


class DataModule(pl.LightningDataModule):
    """
    Data module for handling the dataset.

    This class is responsible for loading and preparing the ASVspoof2024 dataset
    for evaluation. It initializes the dataset, applies transformations, and
    creates DataLoaders for the evaluation stage.

    Args:
        batch_size (int): Batch size for data loading.
        audio_path (str): Path to the directory containing audio files.
        protocol_file_path (str): Path to the protocol file.
        transform (callable, optional): A function/transform to apply to the data.
    """
    def __init__(self, batch_size, protocol_file_path, transform):
        super().__init__()
        self.batch_size = batch_size
        self.protocol_file_path = protocol_file_path
        self.transform = transform

    def prepare_data(self):
        """
        Empty prepare_data method left in intentionally. 
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#prepare-data
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """
        Sets up the dataset for the specified stage.

        This method generates a list of spoofed audio file names from the protocol file
        and initializes the evaluation dataset.

        Args:
            stage (str, optional): The stage for which to set up the dataset. Default is None.
        """
        """
        Method to setup your datasets, here you can use whatever dataset class you have defined in Pytorch and prepare the data in order to pass it to the loaders later
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#setup
        """

        files_eval = generate_filelist(
           metadata_file=self.protocol_file_path,
       )

        if stage == "test":
           self.eval = DatasetEval(
               file_paths=files_eval,
                transform=self.transform
           )

    def test_dataloader(self):
        """
        Creates a DataLoader for the evaluation dataset.

        This method returns a DataLoader for the evaluation dataset (`self.eval`).
        If `self.transform` is set, the `collate_fn` is set to `None`.
        Otherwise, the `collate_fn` is set to `self.collate_fn_eval` to handle
        padding of sequences.

        Returns:
            DataLoader: A DataLoader for the evaluation dataset.
        """
        collate_fn = None if self.transform else self._collate_fn_eval
        return DataLoader(
            dataset=self.eval,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            # num_workers=0
        )
    
    def _collate_fn_eval(self, batch):
       """
        Collate function for evaluation DataLoader.

        This method pads sequences in the batch to the same length and returns
        the padded sequences along with their corresponding utterance IDs.

        Args:
            batch (list of tuples): A batch of data, where each element is a tuple
                                    containing a tensor and an utterance ID.

        Returns:
            tuple: A tuple containing:
                - x (Tensor): A tensor of padded sequences.
                - utt_id (tuple): A tuple of utterance IDs.
        """
       x, file_path = zip(*batch)
       x = torch.nn.utils.rnn.pad_sequence(
           [tensor.squeeze() for tensor in x], batch_first=True, padding_value=0.0
       )
       return x, file_path


