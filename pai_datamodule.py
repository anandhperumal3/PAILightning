import datasets
import json
import os
import pytorch_lightning as pl
import torch
import requests
import warnings
from datasets.arrow_dataset import Dataset
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, TensorDataset
from typing import Any, Optional
from datasets import load_dataset
from transformers import AutoTokenizer


class PrivateAISynthetic(pl.LightningDataModule):
    def __init__(self, data_file, text_feature_name):
        """
        Private-AI Data Module, for synthetic data generation
        :param text_features: list of text feature names in the dataset that needs a synthetic data generation
        :param data_file: path for input data file
        """
        super().__init__()
        self.val_dataset = None
        self.tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
        self.key = 'XXX'
        self.split_size = 0.2 #split_size
        self.save_cache_dataset_dir = './output_dir' #save_dataset_dir
        self.batch_size = 32
        self.test_dataset=None
        self.predict_dataset=None
        self.padding = 'max_length'
        self.truncation = True
        self.batched = True
        self.data_synthetic = False
        self.cache = False
        self.data_file = data_file
        self.text_feature_name = text_feature_name


    def prepare_data(self) -> None:
        """
        This method iterates across the dataset, producing a synthesis phrase for each text in the train, test, and
        predict.The phrase is then converted into a tensor using the HuggingFace tokenizer.
        If cache is set to True, the tensor is saved to the directory; in subsequent calls, prepare data may be avoided
        and setup can be performed directly by specifying the save directory path when initiating the PAIDataModule.

        :return: None
        """

        dataset = load_dataset('csv',data_files=self.data_file)
        self.train_dataset = dataset['train']
        if 'test' in dataset:
            self.test_dataset = dataset['test']
        if 'validation' in dataset:
            self.val_dataset = dataset['validation']


        self.train_dataset = self.train_dataset.map(
        lambda e: self.tokenizer(e[self.text_feature_name], truncation=self.truncation, padding=self.padding),
        batched=self.batched)

        if self.val_dataset:
            self.val_dataset = self.val_dataset.map(
                                lambda e: self.tokenizer(e[self.text_feature_name], truncation=self.truncation, padding=self.padding),
                                batched=self.batched)
        if self.test_dataset:
            self.test_dataset = self.test_dataset.map(
                                    lambda e: self.tokenizer(e[self.text_feature_name], truncation=self.truncation, padding=self.padding),
                                    batched=self.batched)
        if self.predict_dataset:
            self.predict_dataset = self.predict_dataset.map(
                                    lambda e: self.tokenizer(e[self.text_feature_name], truncation=self.truncation, padding=self.padding),
                                    batched=self.batched)


        if self.cache:
            self.save_cache_dataset_dir = self.save_cache_dataset_dir if self.save_cache_dataset_dir else './'
            self.train_dataset.save_to_disk(f"{self.save_cache_dataset_dir}/train_dataset")
            if self.test_dataset:
                self.test_dataset.save_to_disk(f"{self.save_cache_dataset_dir}/test_dataset")
            if self.predict_dataset:
                self.predict_dataset.save_to_disk(f"{self.save_cache_dataset_dir}/predict_dataset")

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Assigning the train/val/test/predict split
        :param stage: it defines which setup logic for which split needs to defined.
        """
        # self.prepare_data()
        if self.cache:
            self.save_cache_dataset_dir = self.save_cache_dataset_dir if self.save_cache_dataset_dir else './'

        if stage == "fit" or stage is None:
            if self.cache:
                assert os.path.isdir(
                    f"{self.save_cache_dataset_dir}/train_dataset"), "train dataset cache is missing, please check the path"
                self.train_dataset = datasets.load_from_disk(f"{self.save_cache_dataset_dir}/train_dataset")
                self.data_synthetic = True
            assert 0 <= self.split_size <= 1.0, "split_size should be in the range of 0 to 1"
            if 0 < self.split_size < 1 and not self.val_dataset:
                dataset = self.train_dataset.train_test_split(test_size=self.split_size)
                self.train_dataset = dataset['train']
                self.val_dataset = dataset['test']
            else:
                self.train_dataset = self.train_dataset

        if stage == "test" or stage is None and self.test_dataset:
            if self.cache:
                assert os.path.isdir(
                    f"{self.save_cache_dataset_dir}/test_dataset"), "test dataset cache is missing, please check the path"
                self.test_dataset = datasets.load_from_disk(f"{self.save_cache_dataset_dir}/test_dataset")
                self.data_synthetic = True

        if stage == "predict" or stage is None and self.predict_dataset:
            if self.cache:
                assert os.path.isdir(
                    f"{self.save_cache_dataset_dir}/predict_dataset"), "predict dataset cache is missing, please check the path"
                self.predict_dataset = datasets.load_from_disk(f"{self.save_cache_dataset_dir}/predict_dataset")
                self.data_synthetic = True

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """
        This function creates a pytorch dataloader for training dataset with the defined batch size
        :return: Pytorch Dataloader
        """
        print(self.train_dataset)
        if not self.data_synthetic:
            warnings.warn("Data is not synthetic, please call prepare_data function for synthetic data generation")
        self.train_dataset = TensorDataset(torch.tensor(self.train_dataset['input_ids'][:5]),
                                           torch.tensor(self.train_dataset['attention_mask'][:5]),
                                           torch.tensor(self.train_dataset['label'][:5]))
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """
        This function creates a pytorch dataloader for validation dataset with the defined batch size
        :return: Pytorch Dataloader
        """
        if not self.data_synthetic:
            warnings.warn("Data is not synthetic, please call prepare_data function for synthetic data generation")
        assert self.val_dataset, "validation dataset is not defined"
        self.val_dataset = TensorDataset(torch.tensor(self.val_dataset['input_ids']),
                                         torch.tensor(self.val_dataset['attention_mask']),
                                         torch.tensor(self.val_dataset['label']))
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """
        This function creates a pytorch dataloader for test dataset with the defined batch size
        :return: Pytorch Dataloader
        """
        if not self.data_synthetic:
            warnings.warn("Data is not synthetic, please call prepare_data function for synthetic data generation")
        assert self.test_dataset, "test dataset is not defined"
        self.test_dataloader = TensorDataset(torch.tensor(self.test_dataloader['input_ids']),
                                             torch.tensor(self.test_dataloader['attention_mask']),
                                             torch.tensor(self.test_dataloader['label']))
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        """
        This function creates a pytorch dataloader for predict dataset with the defined batch size
        :return: Pytorch Dataloader
        """
        if not self.data_synthetic:
            warnings.warn("Data is not synthetic, please call prepare_data function for synthetic data generation")
        assert self.predict_dataset, "predict dataset is not defined"
        self.predict_dataset = TensorDataset(torch.tensor(self.predict_dataset['input_ids']),
                                             torch.tensor(self.predict_dataset['attention_mask']),
                                             torch.tensor(self.predict_dataset['label']))
        return DataLoader(self.predict_dataset, batch_size=self.batch_size)
