# Game-Theory-Agent

An agent built for **Code & Conquer** multi-agent game theory competition at the
Indian Institute of Science (IISc) Open Day 2026. This repo holds our submission, which placed first in the tournament.

## The Game

Two or more commanders start with a fixed pool of soldiers. Over several rounds, each player secretly allocates soldiers across a set of battlefields, each worth a different number of points. Whoever sends the most soldiers to a field in a round wins that field's points (ties give nobody anything). Soldiers spent don't come back, so the real challenge is managing a shrinking budget across rounds while reacting to what opponents have done so far. Full rules are in [`detailed_rules.md`](detailed_rules.md).

## Our Approach: Monte Carlo Simulation

Rather than committing to a fixed formula or a single game-theoretic equilibrium, the agent decides each round's allocation by simulation:

- It generates a large number of candidate allocations for the current round, respecting the remaining budget and number of fields.
- For each candidate, it simulates likely opponent responses (informed by their allocation history so far) and estimates the expected points won.
- It also accounts for the rounds still remaining, so it doesn't overcommit early and get left with too few soldiers to contest high-value fields later.
- The allocation with the best simulated expected outcome is chosen for that round.

This lets the agent adapt as more history becomes available, rather than following a static plan, while still reasoning about the long-run budget rather than just the immediate round.

## Repo Structure

- `agent_class.py` — abstract base class that agents must inherit from
- `env.py` — the game environment
- `validate.py` — checks a submitted agent folder's structure and the values returned by `get_allocation`
- `run_tournament.py` — runs a full tournament between all agents in `Sample_Agents` and my agent
- `human_play.py` — lets a human play interactively against the agents
- `config.toml` — tournament configuration (players, fields, rounds, starting soldiers)
- `detailed_rules.md` — full rules of the game
- `Sample_Agents/` — baseline bots to test against
- `Submission/` — the submission (`your_agent.py`, containing the Monte Carlo agent)

## Running It

With `pip`:
```bash
pip install numpy
python run_tournament.py
```

With `uv`:
```bash
uv run run_tournament.py
```

You can also run `validate.py` first to confirm the agent folder is structured correctly and that `get_allocation` returns valid values.
