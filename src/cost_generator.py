import numpy as np
import torch
from typing import List, Optional, Dict, Any
4

class VanillaCost:
    """Class for contracting object for generating bidding clients

    The contract based method will pay (Payment) the clients until they satisfy (according to the self.cost_clients)
    The utility function is considered based on Accuracy + Cost + Payment as follows:
        - If Accuracy + Cost + Payment > Threshold

    Attributes:

    """
    def __init__(self, clients, hparam):
        self.clients = clients
        self.num_clients = len(self.clients)
        self.alpha = 0.5

    def forward(self):
        bidded_client_indices = torch.ones(self.client_num) / self.alpha
        return bidded_client_indices