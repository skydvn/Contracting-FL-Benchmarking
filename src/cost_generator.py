import numpy as np
import torch
from typing import List, Optional, Dict, Any
4

class vanilla_cost:
    """Class for contracting object for generating bidding new_clients

    The contract based method will pay (Payment) the new_clients until they satisfy (according to the self.cost_clients)
    The utility function is considered based on Accuracy + Cost + Payment as follows:
        - If Accuracy + Cost + Payment > Threshold

    Attributes:

    """
    def __init__(self, new_clients, hparam = None):
        self.new_clients = new_clients
        self.num_clients = len(self.new_clients)
        self.alpha = 0.5

    def forward(self):
        cost_values = torch.ones(self.num_clients) / self.alpha
        return cost_values

    def __call__(self):
        """Allow the object to be called like a function"""
        return self.forward()

class fixed_cost(vanilla_cost):
    def __init__(self, new_clients, hparam = None):
        super().__init__(new_clients, hparam)
        self.fixed_cost = hparam['fixed_cost']

    def forward(self):
        return torch.full((self.num_clients,), self.fixed_cost)