# opponent_aware_agent.py
from agent_class import AbstractAgent
import numpy as np
import math

class Agent(AbstractAgent):
    """
    Estimate opponents' mean bids per field from history and then try to outbid where value/bid ratio is favorable.
    """
    def __init__(self, name="OpponentAware"):
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

        # Build empirical average opponent bid per field across history (exclude self if present)
        if not history:
            # fallback to proportional
            fv = np.array(field_values, dtype=float)
            if fv.sum() == 0:
                return [0]*num_fields
            base = np.floor((fv / fv.sum()) * round_budget).astype(int)
            rem = round_budget - base.sum()
            for i in np.argsort(-(fv - np.floor(fv)))[:rem]:
                base[i] += 1
            return base.tolist()

        # accumulate
        # history is a list of dicts {agent_name: [allocations]}
        # compute mean per field for other agents
        total = np.zeros(num_fields, dtype=float)
        count = 0
        for rnd in history:
            for agent_name, alloc in rnd.items():
                if agent_name == self.name:
                    continue
                total += np.array(alloc, dtype=float)
                count += 1
        if count == 0:
            predicted = np.zeros(num_fields, dtype=float)
        else:
            predicted = total / count  # mean bid per field

        # For each field, compute "bang-for-buck" = value / (predicted + 1)
        fv = np.array(field_values, dtype=float)
        score = fv / (predicted + 1e-6 + 1.0)  # +1 as step to beat predicted
        # sort fields by descending score
        order = np.argsort(-score)

        alloc = [0]*num_fields
        remaining = round_budget

        # greedily attempt to secure highest marginal ROI fields:
        for idx in order:
            if remaining <= 0:
                break
            # attempt to place just above predicted mean: target = ceil(predicted[idx]) + 1
            target = int(np.ceil(predicted[idx])) + 1
            # but if target is too expensive relative to field value, skip and place minimal presence
            # cost effectiveness threshold: only pay up to value / 2 (tunable)
            max_reasonable = math.ceil(fv[idx] / 1.5)  # tuneable parameter
            # clamp target so it is not wasteful
            bid = min(target, max(remaining, target))
            if bid > max_reasonable:
                # don't overpay; place 0 or small presence
                small = 0
                if remaining > 0 and fv[idx] >= 1:
                    small = 1  # single soldier to possibly change tie dynamics (but ties give zero)
                bid = small
            if bid > remaining:
                bid = remaining
            alloc[idx] = int(bid)
            remaining -= alloc[idx]

        # If leftover, place uniformly small amounts on best remaining fields
        if remaining > 0:
            idxs = np.argsort(-fv)
            for i in idxs:
                if remaining <= 0:
                    break
                alloc[i] += 1
                remaining -= 1

        # ensure non-negative ints and sum <= round_budget
        alloc = [int(max(0,a)) for a in alloc]
        # trim if needed
        while sum(alloc) > round_budget:
            # remove from lowest value field allocated > 0
            candidates = [i for i,a in enumerate(alloc) if a>0]
            if not candidates:
                break
            remove_idx = min(candidates, key=lambda i: fv[i])
            alloc[remove_idx] -= 1

        return alloc