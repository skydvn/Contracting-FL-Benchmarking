import numpy as np
import torch
from typing import List, Optional, Dict, Any
4

class VanillaContractor:
    """Class for contracting object for generating bidding clients

    The contract based method will pay (Payment) the clients until they satisfy (according to the self.cost_clients)
    The utility function is considered based on Accuracy + Cost + Payment as follows:
        - If Accuracy + Cost + Payment > Threshold

    Attributes:

    """
    def __init__(self, new_clients, hparam=None):
        self.new_clients = new_clients
        self.num_clients = len(self.new_clients)
        self.client_probability = None
        self.selected_num = 2

    def forward(self, accs, cost_values):
        client_payment = torch.ones(self.num_clients)
        accs = torch.tensor(accs, dtype=torch.float32)

        utilities = client_payment + accs - cost_values
        self.client_probability = torch.softmax(utilities, dim=0)

        top_probs, top_indices = torch.topk(self.client_probability, self.selected_num)
        selected_clients = [self.new_clients[i] for i in top_indices.tolist()]

        # print header
        print("\nClient summary (per client):")
        print(f"{'ID':<5}{'acc':<10}{'cost':<10}{'pay':<8}{'utility':<12}{'prob':<10}")
        print("-" * 55)

        # print each client info in one row
        for client, acc, cost, pay, util, prob in zip(
            self.new_clients,
            accs.tolist(),
            cost_values.tolist(),
            client_payment.tolist(),
            utilities.tolist(),
            self.client_probability.tolist()
        ):
            print(f"{client.client_id:<5}{acc:<10.4f}{cost:<10.4f}{pay:<8.4f}{util:<12.4f}{prob:<10.4f}")

        # print selected clients id
        selected_ids = [client.client_id for client in selected_clients]
        print(f"\nSelected clients: {selected_ids}")

        return selected_clients

    
    def __call__(self, client_val_results, cost_values):
        """Allow the object to be called like a function"""
        return self.forward(client_val_results, cost_values)


class XContractor(VanillaContractor):
    def __init__(self, clients, hparam):
        # gọi constructor của class cha
        super().__init__(clients, hparam)

        # mở rộng thêm thuộc tính riêng của XContractor
        self.clients = clients
        self.num_clients = len(self.clients)
        self.client_probability = None

    def forward(self):
        pass