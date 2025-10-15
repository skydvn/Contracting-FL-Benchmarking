# federated_utils.py
import logging

def setup_wandb_if_needed(args, wandb_project, hparam):
    if not args.no_wandb:
        import wandb
        wandb.init(project=wandb_project, entity="your_entity", config=hparam)
        wandb.run.log_code()
        hparam['wandb'] = True
    else:
        hparam['wandb'] = False

def initialize_clients(client_method_name, num_clients, device, training_datasets, ds_bundle, hparam, id_offset=0):
    clients = []
    for k in range(num_clients):
        client_id = k + id_offset
        client = eval(client_method_name)(client_id, device, training_datasets[k], ds_bundle, hparam)
        clients.append(client)
    return clients

def initialize_server(server_method_name, device, ds_bundle, hparam, global_dataloader=None):
    central_server = eval(server_method_name)(device, ds_bundle, hparam)
    if server_method_name == "FedDG" and global_dataloader is not None:
        central_server.set_amploader(global_dataloader)
    return central_server

def run_initial_federated_training(central_server):
    central_server.fit()

def run_expansion_cycle(central_server, known_clients, new_clients, contract_method_name, cost_method_name):
    # create contractor & cost generator
    contractor = eval(contract_method_name)(new_clients)
    cost_generator = eval(cost_method_name)(new_clients)

    client_val_results = []
    for client in new_clients:
        trial_clients = known_clients + [client]
        central_server.register_clients(trial_clients)
        val_acc = central_server.trial_fit(num_trial_rounds=1)
        client_val_results.append({'client': client, 'acc': val_acc})
        print(f"Trial with client {client.client_id} finished with validation score: {val_acc:.4f}")

    accs = [item["acc"] for item in client_val_results]
    cost_values = cost_generator()
    selected_clients = contractor(accs, cost_values)

    clients_to_register = known_clients + selected_clients
    return clients_to_register
