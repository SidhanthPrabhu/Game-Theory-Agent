# montecarlo_agent.py
from agent_class import AbstractAgent
import numpy as np
import random
import math
from collections import defaultdict

class Agent(AbstractAgent):
    """
    Monte Carlo sampling of opponent bids (from history) and sampling candidate allocations.
    """
    def __init__(self, name="MonteCarlo", sims=150, candidates=60):
        super().__init__(name)
        self.sims = sims
        self.candidates = candidates

    def _build_empirical_field_distributions(self, history, num_fields):
        # returns list of lists: per-field list of observed bids (across opponents & rounds)
        per_field = [[] for _ in range(num_fields)]
        if not history:
            # uniform low samples
            for f in range(num_fields):
                per_field[f] = [0,1,2]
            return per_field

        for rnd in history:
            for agent_name, alloc in rnd.items():
                if agent_name == self.name:
                    continue
                for f, v in enumerate(alloc):
                    per_field[f].append(int(v))
        # fallback: if some fields have no data, add small values
        for f in range(num_fields):
            if not per_field[f]:
                per_field[f] = [0,1,2]
        return per_field

    def _sample_opponent_profile(self, per_field, num_other_agents):
        # sample for each opponent a full-array by independently sampling each field
        # this assumes independence; crude but often works
        sampled_allocs = []
        for _ in range(num_other_agents):
            a = [int(random.choice(per_field[f])) for f in range(len(per_field))]
            sampled_allocs.append(a)
        return sampled_allocs

    def _score_allocation(self, my_alloc, sampled_others, field_values):
        # returns points I get in a single simulated round
        num_fields = len(field_values)
        points = 0
        for f in range(num_fields):
            bids = [sample[f] for sample in sampled_others] + [my_alloc[f]]
            max_bid = max(bids)
            winners = [i for i,b in enumerate(bids) if b==max_bid]
            # my index is last in the list
            if len(winners)==1 and winners[-1]==len(bids)-1:
                points += field_values[f]
        return points

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
            # be slightly conservative early on
            round_budget = max(1, int(current_balance / rounds_left))

        per_field = self._build_empirical_field_distributions(history, num_fields)

        # number of other agents from balances (includes self)
        num_players = len(balances)
        num_other_agents = max(0, num_players - 1)

        # generate candidate allocations (randomized around proportional and top-k)
        candidates = []
        fv = np.array(field_values, dtype=float)
        # base proportional candidate
        if fv.sum() > 0:
            base = np.floor((fv / fv.sum()) * round_budget).astype(int)
        else:
            base = np.zeros(num_fields, dtype=int)
        # add some jittered candidates
        for _ in range(self.candidates):
            cand = base.copy()
            # jitter: randomly move up to 30% of round_budget around
            jitter = random.randint(0, max(1, int(round_budget * 0.3)))
            for _ in range(jitter):
                i = random.randrange(num_fields)
                j = random.randrange(num_fields)
                # move 1 soldier from j to i (if possible)
                if cand[j] > 0:
                    cand[j] -= 1
                    cand[i] += 1
            # also sometimes concentrate on top fields
            if random.random() < 0.3:
                topk = max(1, int(0.3 * num_fields))
                top_idx = np.argsort(-fv)[:topk]
                # shift some budget to topk
                shift = random.randint(0, max(0, round_budget//4))
                for _ in range(shift):
                    from_idx = random.randrange(num_fields)
                    to_idx = int(random.choice(top_idx))
                    if cand[from_idx] > 0:
                        cand[from_idx] -= 1
                        cand[to_idx] += 1
            # ensure sum equals round_budget (fill or trim)
            s = int(cand.sum())
            if s < round_budget:
                # add remaining to highest value fields
                idxs = np.argsort(-fv)
                rem = round_budget - s
                for ii in idxs:
                    if rem <= 0:
                        break
                    cand[ii] += 1
                    rem -= 1
            elif s > round_budget:
                # remove excess from lowest value allocated
                rem = s - round_budget
                while rem > 0:
                    pos = np.argmin([fv[i] if cand[i]>0 else float('inf') for i in range(num_fields)])
                    if cand[pos] > 0:
                        cand[pos] -= 1
                        rem -= 1
                    else:
                        break
            candidates.append(cand.tolist())

        # Monte Carlo evaluate candidates
        best = candidates[0]
        best_score = -1.0
        sims = max(20, min(self.sims, 400))  # keep sims bounded
        for cand in candidates:
            total_pts = 0.0
            for _ in range(sims):
                # sample opponents
                sampled_others = []
                # if no opponent data, sample zeros
                if num_other_agents == 0:
                    sampled_others = []
                else:
                    for _ in range(num_other_agents):
                        sampled = [int(random.choice(per_field[f])) for f in range(num_fields)]
                        sampled_others.append(sampled)
                pts = self._score_allocation(cand, sampled_others, field_values)
                total_pts += pts
            exp = total_pts / sims
            if exp > best_score:
                best_score = exp
                best = cand

        return [int(x) for x in best]