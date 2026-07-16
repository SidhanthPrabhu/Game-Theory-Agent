from agent_class import AbstractAgent
import numpy as np
import random


class Agent(AbstractAgent):

    """ How much to spend this round """
    def compute_budget(self, balance, total_rounds, current_round, balances):
        rounds_left = total_rounds - current_round + 1
        if rounds_left == 1:
            return balance
        base = balance / rounds_left

        # Calculate how aggresive / conservative to spend depending on opponents' budget
        opp_balances = [b for n, b in balances.items() if n != self.name]
        avg_opp = np.mean(opp_balances) if opp_balances else balance
        pressure = avg_opp / (balance + 1)
        if pressure < 0.5:
            factor = 1.2    # ahead - spend more to stay ahead
        elif pressure < 1.2:
            factor = 1.0    # roughly even
        else:
            factor = 0.85   # behind - conserve slightly
        budget = int(base * factor)

        # always keep at least 1 soldier per remaining round
        return max(1, min(budget, balance - (rounds_left - 1)))


    """ Simulate expected score - to beat all opponents """
    def simulate(self, alloc, opp_history, values, num_opponents, sims=100):
        num_fields = len(values)
        score = 0
        for i in range(sims):
            # Sample one bid per opponent, take the worst case (max) per field
            opp_max = np.zeros(num_fields)
            for j in range(num_opponents):
                if opp_history:
                    opp_vec = np.array(random.choice(opp_history))
                else:
                    opp_vec = np.zeros(num_fields)
                opp_max = np.maximum(opp_max, opp_vec)
            for k in range(num_fields):
                if alloc[k] > opp_max[k]:
                    score += values[k]

        return score / sims


    """ Generate a pool of candidate allocations to try """
    def generate_candidates(self, budget, values, num_fields):
        candidates = []
        v = np.array(values, dtype=float)

        # 1. Proportional to field value
        alloc = np.floor((v / v.sum()) * budget).astype(int)
        alloc[0] += budget - alloc.sum()
        candidates.append(alloc)

        # 2. All-in on best field
        alloc = np.zeros(num_fields, dtype=int)
        alloc[np.argmax(v)] = budget
        candidates.append(alloc)

        # 3. Split between top 2 fields
        top2 = np.argsort(-v)[:2]
        for r in [0.3, 0.5, 0.7]:
            alloc = np.zeros(num_fields, dtype=int)
            alloc[top2[0]] = int(budget * r)
            alloc[top2[1]] = budget - alloc[top2[0]]
            candidates.append(alloc)

        # 4. Random splits focused on top half of fields
        topk = np.argsort(-v)[:max(1, num_fields // 2)]
        for _ in range(15):
            alloc = np.zeros(num_fields, dtype=int)
            rem = budget
            for i in topk[:-1]:
                r = random.randint(0, rem)
                alloc[i] = r
                rem -= r
            alloc[topk[-1]] = rem
            candidates.append(alloc)

        return candidates

    def get_allocation(
        self,
        current_balance,
        field_values,
        num_fields,
        history,
        balances,
        total_rounds,
        current_round,
    ):
        if current_balance <= 0:
            return [0] * num_fields
        budget = self.compute_budget(
            current_balance, total_rounds, current_round, balances
        )
        # Collect all past opponent allocations as reference
        opp_history = []
        for r in history:
            for agent, alloc in r.items():
                if agent != self.name:
                    opp_history.append(alloc)
        
        # How many opponents are we facing?
        num_opponents = max(1, len(balances) - 1)
        
        # Scale sims with game phase - save compute early
        if current_round <= 2:
            sims = 60
        elif current_round <= total_rounds * 0.6:
            sims = 100
        else:
            sims = 150

        # Pick the candidate with the best simulated score
        candidates = self.generate_candidates(budget, field_values, num_fields)
        best, best_score = candidates[0], -1
        
        for c in candidates:
            s = self.simulate(c, opp_history, field_values, num_opponents, sims)
            if s > best_score:
                best_score = s
                best = c

        return best.astype(int).tolist()