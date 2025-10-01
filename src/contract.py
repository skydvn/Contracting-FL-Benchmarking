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
    def __init__(self, clients, hparam=None):
        self.clients = clients
        self.num_clients = len(self.clients)
        self.client_probability = None
        # TODO We need to define the important variables in Contractor (according to different contract)
        # self.param_list = {}

    def forward(self):
        # Start with all clients not selected
        bidded_client_indices = torch.zeros(self.client_num)

        if self.client_probability is None:
            # No probabilities -> all clients bid
            bidded_client_indices[:] = 1.0
        else:
            # Draw random values for each client
            random_values = torch.rand(self.client_num)
            # Select clients where random < probability
            bidded_client_indices = (random_values < self.client_probability).float()

        return bidded_client_indices
    
    def __call__(self):
        """Allow the object to be called like a function"""
        return self.forward()


class XContractor(VanillaContractor):
    def __init__(self, clients, hparam):
        self.clients = clients
        self.num_clients = len(self.clients)
        self.client_probability = None

    def _client_probability(self):
        # TODO Generate self.client_probability to accept offer
        # TODO Generate List of client probability: [0.8, 0.9, .... , 0.7]
        pass