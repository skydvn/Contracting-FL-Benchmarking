# data_utils.py
import os
from torch.utils.data import DataLoader, RandomSampler

import src.datasets as my_datasets
from src.splitter import *
from wilds import get_dataset

def load_dataset_and_bundle(hparam):
    ds_name = hparam['dataset'].lower()
    path = hparam['dataset_path']
    if ds_name == 'pacs':
        dataset = my_datasets.PACS(version='1.0', root_dir=path, download=True)
    elif ds_name == 'officehome':
        dataset = my_datasets.OfficeHome(version='1.0', root_dir=path, download=True, split_scheme=hparam.get("split_scheme"))
    elif ds_name == 'femnist':
        dataset = my_datasets.FEMNIST(version='1.0', root_dir=path, download=True)
    elif ds_name == 'celeba':
        dataset = get_dataset(dataset="celebA", root_dir=path, download=True)
    else:
        dataset = get_dataset(dataset=ds_name, root_dir=path, download=True)

    # dataset-specific wrapper
    if hparam['client_method'] == "FedSR":
        ds_bundle = eval(hparam["dataset"])(dataset, probabilistic=True)
    else:
        ds_bundle = eval(hparam["dataset"])(dataset, probabilistic=False)

    return dataset, ds_bundle

def prepare_total_subset_and_testloaders(dataset, ds_bundle, hparam):
    if hparam['server_method'] == "FedDG":
        ds_lower = hparam["dataset"].lower()
        if ds_lower == "iwildcam":
            dataset = my_datasets.FourierIwildCam(root_dir=hparam['dataset_path'], download=True)
        elif ds_lower == "pacs":
            dataset = my_datasets.FourierPACS(root_dir=hparam['dataset_path'], download=True, split_scheme=hparam.get("split_scheme"))
        elif ds_lower == "celeba":
            dataset = my_datasets.FourierCelebA(root_dir=hparam['dataset_path'], download=True, split_scheme=hparam.get("split_scheme"))
        elif ds_lower == "camelyon17":
            dataset = my_datasets.FourierCamelyon17(root_dir=hparam['dataset_path'], download=True, split_scheme=hparam.get("split_scheme"))
        elif ds_lower == "femnist":
            dataset = my_datasets.FourierFEMNIST(root_dir=hparam['dataset_path'], download=True, split_scheme=hparam.get("split_scheme"))
        else:
            raise NotImplementedError
        total_subset = dataset.get_subset('train', transform=ds_bundle.test_transform)
    else:
        total_subset = dataset.get_subset('train', transform=ds_bundle.train_transform)

    # build test loaders
    testloader = {}
    for split in dataset.split_names:
        if split != 'train':
            ds = dataset.get_subset(split, transform=ds_bundle.test_transform)
            dl = get_eval_loader(loader='standard', dataset=ds, batch_size=hparam["batch_size"])
            testloader[split] = dl

    return total_subset, testloader

def make_global_dataloader(total_subset, hparam):
    sampler = RandomSampler(total_subset, replacement=True)
    global_dataloader = DataLoader(total_subset, batch_size=hparam["batch_size"], sampler=sampler)
    return global_dataloader

def get_training_datasets(total_subset, num_shards, iid, seed, ds_bundle):
    if num_shards == 1:
        return [total_subset]
    elif num_shards > 1:
        return NonIIDSplitter(num_shards=num_shards, iid=iid, seed=seed).split(
            total_subset, ds_bundle.groupby_fields, transform=ds_bundle.train_transform
        )
    else:
        raise ValueError("num_shards should be >= 1")
