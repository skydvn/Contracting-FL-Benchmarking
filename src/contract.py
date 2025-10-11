import numpy as np
import torch
from typing import List, Optional, Dict, Any
4

class vanilla_contractor:
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


class random_contractor(vanilla_contractor):
    def __init__(self, clients, hparam=None):
        super().__init__(clients, hparam)

    def forward(self, accs, cost_values, payment_values):
        print(f"Randomly selecting {self.selected_num} clients...")
        self.client_probability = torch.rand(self.num_clients)
        top_probs, top_indices = torch.topk(self.client_probability, self.selected_num)
        selected_clients = [self.new_clients[i] for i in top_indices.tolist()]
        return selected_clients
    
class greedy_contractor(vanilla_contractor):
    def __init__(self, new_clients, hparam=None):
        super().__init__(new_clients, hparam)
        self.budget = hparam.get('budget', 2.0)

    def forward(self, accs, cost_values, payment_values):
        accs = torch.tensor(accs, dtype=torch.float32) if not torch.is_tensor(accs) else accs.float()
        cost_values = torch.tensor(cost_values, dtype=torch.float32) if not torch.is_tensor(cost_values) else cost_values.float()
        self.client_payment = torch.tensor(payment_values, dtype=torch.float32) if not torch.is_tensor(payment_values) else payment_values.float()

        utilities = accs + self.client_payment - cost_values

        safe_cost = cost_values.clone()
        safe_cost[safe_cost == 0] = 1e-6
        ratio = utilities / safe_cost
        sorted_indices = torch.argsort(ratio, descending=True)

        selected_clients = []
        total_cost = 0.0

        for idx in sorted_indices.tolist():
            client_cost = cost_values[idx].item()
            if total_cost + client_cost <= self.budget:
                selected_clients.append(self.new_clients[idx])
                total_cost += client_cost

        self._print_summary(accs, cost_values, utilities, selected_clients)

        return selected_clients

    def _print_summary(self, accs, cost_values, utilities, selected_clients):
        print("\nGreedy Contractor Summary:")
        print(f"{'ID':<5}{'acc':<8}{'cost':<8}{'pay':<8}{'utility':<10}")
        print("-"*45)
        for client, acc, cost, pay, util in zip(
            self.new_clients,
            accs.tolist(),
            cost_values.tolist(),
            self.client_payment.tolist(),
            utilities.tolist()
        ):
            print(f"{client.client_id:<5}{acc:<8.4f}{cost:<8.4f}{pay:<8.4f}{util:<10.4f}")

        selected_ids = [c.client_id for c in selected_clients]
        print(f"\nSelected clients (Greedy) under budget {self.budget}: {selected_ids}")

    def __call__(self, accs, cost_values, payment_values=None):
        return self.forward(accs, cost_values, payment_values)
    
class knapsack_contractor(vanilla_contractor):
    def __init__(self, new_clients, hparam=None):
        super().__init__(new_clients, hparam)
        self.budget = hparam.get('budget', 2.0)

    def forward(self, accs, cost_values, payment_values):
        accs = torch.tensor(accs, dtype=torch.float32) if not torch.is_tensor(accs) else accs.float()
        cost_values = torch.tensor(cost_values, dtype=torch.float32) if not torch.is_tensor(cost_values) else cost_values.float()
        client_payment = torch.tensor(payment_values, dtype=torch.float32) if not torch.is_tensor(payment_values) else payment_values.float()

        utilities = accs + client_payment - cost_values

        scale = 1
        int_costs = (cost_values * scale).int()
        int_budget = int(self.budget * scale)
        dp = torch.zeros((self.num_clients + 1, int_budget + 1))

        # Fill DP table
        for i in range(1, self.num_clients + 1):
            for w in range(int_budget + 1):
                if int_costs[i-1] <= w:
                    dp[i][w] = max(dp[i-1][w], dp[i-1][w - int_costs[i-1]] + utilities[i-1])
                else:
                    dp[i][w] = dp[i-1][w]

        # Traceback
        w = int_budget
        selected_clients = []
        for i in range(self.num_clients, 0, -1):
            if dp[i][w] != dp[i-1][w]:
                selected_clients.append(self.new_clients[i-1])
                w -= int_costs[i-1]

        selected_clients.reverse()

        self._print_summary(accs, cost_values, utilities, selected_clients)

        return selected_clients

    def _print_summary(self, accs, cost_values, utilities, selected_clients):
        print("\nKnapsack Contractor Summary:")
        print(f"{'ID':<5}{'acc':<8}{'cost':<8}{'pay':<8}{'utility':<10}")
        print("-"*45)
        for client, acc, cost, pay, util in zip(
            self.new_clients,
            accs.tolist(),
            cost_values.tolist(),
            self.client_payment.tolist(),
            utilities.tolist()
        ):
            print(f"{client.client_id:<5}{acc:<8.4f}{cost:<8.4f}{pay:<8.4f}{util:<10.4f}")

        selected_ids = [c.client_id for c in selected_clients]
        print(f"\nSelected clients (Knapsack) under budget {self.budget}: {selected_ids}")

    def __call__(self, accs, cost_values, payment_values=None):
        return self.forward(accs, cost_values, payment_values)
