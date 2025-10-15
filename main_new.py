# main.py
import json
import os
import time
import gc
import logging
import argparse

import torch
import wandb
from tqdm import tqdm

from utils.data_utils import (
    load_dataset_and_bundle,
    prepare_total_subset_and_testloaders,
    make_global_dataloader,
    get_training_datasets
)
from utils.federated_utils import (
    setup_wandb_if_needed,
    initialize_clients,
    initialize_server,
    run_initial_federated_training,
    run_expansion_cycle
)
from utils.utils import set_seed, mkdirs_if_needed

WANDB_PROJECT = "your_project"
WANDB_ENTITY = "your_entity"

def main(args):
    # load config and hparams
    with open(args.config_file) as fh:
        config = json.load(fh)
    hparam = vars(args)
    hparam.update(config)

    # wandb setup
    wandb_project = WANDB_PROJECT + '_' + hparam['dataset']
    setup_wandb_if_needed(args, wandb_project, hparam)

    # device, seed, dirs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    set_seed(hparam['seed'])
    mkdirs_if_needed(hparam['data_path'], ["opt_dict", "sch_dict", "models"])

    # optimizer preprocessing (kept same logic)
    if hparam['optimizer'] == 'torch.optim.SGD':
        hparam['optimizer_config'] = {'lr':hparam['lr'], 'momentum': hparam['momentum'], 'weight_decay': hparam['weight_decay']}
    elif hparam['optimizer'] in ('torch.optim.Adam', 'torch.optim.AdamW'):
        hparam['optimizer_config'] = {'lr':hparam['lr'], 'eps': hparam['eps'], 'weight_decay': hparam['weight_decay']}

    # dataset / transforms / ds_bundle
    dataset, ds_bundle = load_dataset_and_bundle(hparam)

    # total subset and test loaders (handles Fourier variants inside)
    total_subset, testloader = prepare_total_subset_and_testloaders(dataset, ds_bundle, hparam)

    # global dataloader (sampler replacement)
    global_dataloader = make_global_dataloader(total_subset, hparam)

    # split into client shards
    training_datasets = get_training_datasets(
        total_subset,
        num_shards=hparam['num_clients'],
        iid=hparam.get('iid', True),
        seed=hparam['seed'],
        ds_bundle=ds_bundle
    )

    # initialize clients
    new_clients = initialize_clients(hparam['client_method'], hparam["num_clients"], device, training_datasets, ds_bundle, hparam)

    logging.info("successfully initialized new clients!")
    gc.collect()

    # initialize server and register
    central_server = initialize_server(hparam['server_method'], device, ds_bundle, hparam, global_dataloader)
    if hparam['start_epoch'] == 0:
        central_server.setup_model(None, 0)
    else:
        central_server.setup_model(hparam['resume_file'], hparam['start_epoch'])

    central_server.register_clients(new_clients)
    central_server.register_testloader(testloader)

    # initial federated training
    run_initial_federated_training(central_server)

    # Contract -> Expansion cycle (modularized)
    known_clients = list(new_clients)
    # Expand: create new client objects with offset ids
    offset = len(known_clients)
    new_clients = initialize_clients(hparam['client_method'], hparam["num_clients"], device, training_datasets, ds_bundle, hparam, id_offset=offset)

    # contractor & cost generator instantiation and trial fits happen inside run_expansion_cycle
    clients_to_register = run_expansion_cycle(
        central_server,
        known_clients,
        new_clients,
        hparam['contract_method'],
        hparam['cost_method']
    )

    # continue federated learning with selected clients
    central_server.register_clients(clients_to_register)
    central_server.fit(first_time=False)

    hparam["expand_time"] = hparam.get("expand_time", 0) - 1
    print("\n--- Expansion cycle complete ---")

    logging.info("...done all learning process! ...exit program!")
    time.sleep(3)
    exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FedDG Benchmark')
    parser.add_argument('--config_file', help='config file', default="config.json")
    parser.add_argument('--no_wandb', default=True, action="store_true")
    parser.add_argument('--seed', default=1001, type=int)
    parser.add_argument('--num_clients', default=5, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--iid', default=1, type=float)
    parser.add_argument('--server_method', default='FedAvg')
    parser.add_argument('--fraction', default=1, type=float)
    parser.add_argument('--f', default=10, type=int)
    parser.add_argument('--num_rounds', default=3, type=int)
    parser.add_argument('--dataset', default='PACS')
    parser.add_argument('--split_scheme', default='official')
    parser.add_argument('--client_method', default='ERM')
    parser.add_argument('--local_epochs', default=1, type=int)
    parser.add_argument('--n_groups_per_batch', default=2, type=int)
    parser.add_argument('--optimizer', default='torch.optim.Adam')
    parser.add_argument('--lr', default=3e-5, type=float)
    parser.add_argument('--momentum', default=0, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--hparam1', default=1, type=float, help="irm: lambda; rex: lambda; fish: meta_lr; mixup: alpha; mmd: lambda; coral: lambda; groupdro: groupdro_eta; fedprox: mu; feddg: ratio; fedadg: alpha; fedgma: mask_threshold; fedsr: l2_regularizer;")
    parser.add_argument('--hparam2', default=1, type=float, help="fedsr: cmi_regularizer; irm: penalty_anneal_iters; fedadg: second_local_epochs")
    parser.add_argument('--hparam3', default=0, type=float)
    parser.add_argument('--hparam4', default=0, type=float)
    parser.add_argument('--hparam5', default=0, type=float)
    parser.add_argument('--contract_method', default='VanillaContractor')
    parser.add_argument('--cost_method', default='VanillaCost')
    parser.add_argument('--expand_time', default=2, type=int)

    args = parser.parse_args()
    main(args)