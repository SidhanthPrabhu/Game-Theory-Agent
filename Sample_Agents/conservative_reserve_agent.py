# conservative_reserve_agent.py
from agent_class import AbstractAgent
import numpy as np

class Agent(AbstractAgent):
    """
    Keep a fraction of budget in reserve; spend aggressively near end.
    """
    def __init__(self, reserve_fraction=0.25, name="ConservativeReserve"):
        super().__init__(name)
        self.reserve_fraction = reserve_fraction

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

        # target reserve based on starting budget estimate (fall back to fraction of current_balance)
        # simple: keep min(reserve_fraction*start, current_balance/2) - but we don't know start; assume reserve_fraction of current_balance
        reserve_target = int(current_balance * self.reserve_fraction)

        # on final rounds (last 1–2), reduce reserve requirement
        rounds_left = total_rounds - current_round + 1
        if rounds_left <= 2:
            reserve_target = 0

        usable = max(0, current_balance - reserve_target)
        if usable == 0:
            return [0]*num_fields

        fv = np.array(field_values, dtype=float)
        if fv.sum() == 0:
            return [0]*num_fields

        raw = fv * (usable / fv.sum())
        base = np.floor(raw).astype(int)
        rem = int(usable - base.sum())
        if rem > 0:
            frac = raw - np.floor(raw)
            idxs = np.argsort(-frac)
            for i in idxs[:rem]:
                base[i] += 1

        # ensure total <= usable
        alloc = base.tolist()
        while sum(alloc) > usable:
            # drop from smallest value field with >0
            idxs = [i for i,a in enumerate(alloc) if a>0]
            if not idxs:
                break
            smallest = min(idxs, key=lambda i: fv[i])
            alloc[smallest] -= 1

        return alloc