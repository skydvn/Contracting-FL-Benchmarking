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

        return selected_clients
    
    def __call__(self, client_val_results, cost_values):
        """Allow the object to be called like a function"""
        return self.forward(client_val_results, cost_values)


class XContractor(VanillaContractor):
    def __init__(self, clients, hparam):
        self.clients = clients
        self.num_clients = len(self.clients)
        self.client_probability = None

    def _client_probability(self):
        # TODO Generate self.client_probability to accept offer
        # TODO Generate List of client probability: [0.8, 0.9, .... , 0.7]
        pass