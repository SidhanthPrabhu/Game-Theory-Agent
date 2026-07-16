# hybrid_agent.py
from agent_class import AbstractAgent
import numpy as np
import random

class Agent(AbstractAgent):
    """
    Opponent-aware bidding, but randomize some fraction to reduce predictability.
    """
    def __init__(self, rand_frac=0.25, name="Hybrid"):
        super().__init__(name)
        self.rand_frac = rand_frac

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

        # base: opponent-aware as in Agent 3 (simple version)
        # reuse simple proportional fallback if no data
        if current_balance <= 0:
            return [0]*num_fields

        rounds_left = total_rounds - current_round + 1
        if current_round == total_rounds:
            round_budget = current_balance
        else:
            round_budget = max(1, current_balance // rounds_left)

        fv = np.array(field_values, dtype=float)
        base_alloc = [0]*num_fields
        if not history:
            if fv.sum() > 0:
                base_alloc = np.floor((fv / fv.sum()) * round_budget).astype(int).tolist()
            else:
                base_alloc = [0]*num_fields
        else:
            # compute opponents' mean bids
            total = np.zeros(num_fields)
            cnt = 0
            for rnd in history:
                for name, alloc in rnd.items():
                    if name == self.name:
                        continue
                    total += np.array(alloc)
                    cnt += 1
            predicted = total / (cnt if cnt>0 else 1)
            # target: slightly beat predicted on top ROI fields
            roi = fv / (predicted + 1.0)
            order = np.argsort(-roi)
            remaining = round_budget
            for idx in order:
                if remaining<=0:
                    break
                want = int(np.ceil(predicted[idx]))+1
                # cap by value
                want = min(want, int(max(1, fv[idx])))
                if want > remaining:
                    want = remaining
                base_alloc[idx] = want
                remaining -= want
            # if leftover, distribute proportionally to fv
            if remaining>0 and fv.sum()>0:
                add = np.floor((fv / fv.sum()) * remaining).astype(int)
                for i in range(num_fields):
                    base_alloc[i] += int(add[i])
                rem2 = remaining - int(add.sum())
                for i in np.argsort(-fv)[:rem2]:
                    base_alloc[i] += 1

        # randomize a small fraction: move up to rand_frac*round_budget soldiers around
        alloc = base_alloc.copy()
        perturb = int(round(self.rand_frac * round_budget))
        for _ in range(perturb):
            # move one soldier from a random field (with >0) to another random field
            donors = [i for i,a in enumerate(alloc) if a>0]
            if not donors:
                break
            d = random.choice(donors)
            r = random.randrange(num_fields)
            if d != r:
                alloc[d] -= 1
                alloc[r] += 1

        # safety trim
        s = sum(alloc)
        while s > round_budget:
            for i in np.argsort(alloc):
                if alloc[i] > 0:
                    alloc[i] -= 1
                    s -= 1
                    if s <= round_budget:
                        break

        return [int(max(0,a)) for a in alloc]