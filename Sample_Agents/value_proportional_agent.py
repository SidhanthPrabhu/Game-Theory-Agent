# value_proportional_agent.py
from agent_class import AbstractAgent
import numpy as np

class Agent(AbstractAgent):
    """
    Spend roughly budget/(remaining rounds) and split proportionally to field values.
    """
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

        # If nothing left, return zeros
        if current_balance <= 0:
            return [0] * num_fields

        # If final round, spend everything
        rounds_left = total_rounds - current_round + 1
        if current_round == total_rounds:
            round_budget = current_balance
        else:
            round_budget = max(0, current_balance // rounds_left)

        fv = np.array(field_values, dtype=float)
        total = fv.sum()
        if total == 0 or round_budget == 0:
            return [0] * num_fields

        # proportional split (floor), then distribute leftover by fractional parts
        raw = fv * (round_budget / total)
        base = np.floor(raw).astype(int)
        leftover = int(round_budget - base.sum())

        if leftover > 0:
            frac = raw - np.floor(raw)
            idx = np.argsort(-frac)  # descending
            for i in idx[:leftover]:
                base[i] += 1

        # safety: if for some reason we over-allocated (rounding), trim from smallest value fields
        alloc = base.tolist()
        while sum(alloc) > round_budget:
            # reduce one from smallest fv that is >0
            idxs = [i for i,a in enumerate(alloc) if a>0]
            if not idxs:
                break
            smallest = min(idxs, key=lambda i: fv[i])
            alloc[smallest] -= 1

        return alloc