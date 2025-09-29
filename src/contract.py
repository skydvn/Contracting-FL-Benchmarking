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
    def __init__(self, clients, hparam):
        self.clients = clients
        self.num_clients = len(self.clients)
        self.client_probability = None

    def forward(self):
        if self.client_probability is None:
            bidded_client_indices = torch.ones(self.client_num)
        else:
            # TODO This one should be fixed in the future
            # TODO Compare with the self.utility: higher self.utility = higher probability
            # TODO Based on the self.client_probability, randomize and choose if bidded clients are chosen
            # Sample clients based on probability distribution
            # Higher utility clients have higher chance of being selected
            random_values = torch.rand(self.num_clients)
            selected_mask = random_values < self.client_probability
            bidded_client_indices[selected_mask] = 1.0

        return bidded_client_indices


class XContractor(VanillaContractor):
    def __init__(self, clients, hparam):
        self.clients = clients
        self.num_clients = len(self.clients)
        self.client_probability = None

    def _client_probability(self):
        # TODO Generate self.client_probability
        pass