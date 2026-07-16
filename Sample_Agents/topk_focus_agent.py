# topk_focus_agent.py
from agent_class import AbstractAgent
import numpy as np

class Agent(AbstractAgent):
    """
    Concentrate resources on top-k fields (by value). Small randomization to avoid perfect predictability.
    """
    def __init__(self, name="TopK"):
        super().__init__(name)

    def get_allocation(
        self,
        current_balance,
        field_values,
        num_fields,
        history,
        balances,
        total_rounds,
        current_round,
    ) -> list:

        if current_balance <= 0:
            return [0]*num_fields

        rounds_left = total_rounds - current_round + 1
        if current_round == total_rounds:
            round_budget = current_balance
        else:
            round_budget = max(1, current_balance // rounds_left)

        fv = np.array(field_values, dtype=float)
        # choose k = max(1, ceil(30-50% of fields))
        k = max(1, int(np.ceil(0.4 * num_fields)))
        top_idx = np.argsort(-fv)[:k]

        alloc = [0]*num_fields
        # allocate 90% of budget to top-k proportionally, keep 10% as small presence on others
        top_budget = int(round_budget * 0.9)
        rest_budget = round_budget - top_budget

        if fv[top_idx].sum() > 0:
            top_share = (fv[top_idx] / fv[top_idx].sum()) * top_budget
            top_base = np.floor(top_share).astype(int)
            rem = top_budget - top_base.sum()
            if rem > 0:
                frac = top_share - np.floor(top_share)
                idxs = np.argsort(-frac)
                for i in idxs[:rem]:
                    top_base[i] += 1
            # map back
            for j, idx in enumerate(top_idx):
                alloc[idx] = int(top_base[j])
        else:
            # equal split among top
            for idx in top_idx:
                alloc[idx] = top_budget // k

        # spread small presence to others
        other_idx = [i for i in range(num_fields) if i not in top_idx]
        if other_idx and rest_budget > 0:
            each = rest_budget // len(other_idx)
            for idx in other_idx:
                alloc[idx] = each
            leftover = rest_budget - each*len(other_idx)
            for i in range(leftover):
                alloc[other_idx[i]] += 1

        # safety cap (sum <= round_budget)
        total_alloc = sum(alloc)
        while total_alloc > round_budget:
            # reduce 1 from smallest allocated field (prefer non-top)
            candidates = [i for i,a in enumerate(alloc) if a>0]
            smallest = min(candidates, key=lambda i: fv[i])
            alloc[smallest] -= 1
            total_alloc -= 1

        return alloc