# mcts-llm

## MCTSr

Based on [Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B](https://arxiv.org/abs/2406.07394) by Zhang, et al.

At a high level, MCTSr iteratively generates solutions to a specified (math) problem.

In a MCTSr tree, nodes correspond to attempted answers, and edges correspond to attempts to improve the answer.


### Initialize
Generate a solution to the problem. This is generally a "dummy" solution (e.g. "I don't know").

### Select a node to expand
We gather a set of candidate nodes which haven't been fully expanded.

A node is fully expanded if either:
a) it has `max_children`
b) any of its children have a Q value which is greater than its own

Once we've gathered the candidates, we compute UCT scores for each candidate node.
There are a few ways we can make our selection:
1. Greedily (choose the node with the highest UCT)
2. Importance sampling (sample from the set of candidates, weighted by their UCT score)
3. Pairwise importance sampling (sample the max from a pair of nodes from the set of candidates, weighted by the difference between the pair's UCT scores)

### Expand the node

Expansion involves several steps:
1. Generate a critique of the current solution.
2. Refine the solution based on the critique.
3. Add a new child, corresponding to the refined solution.
4. Self-evaluate the `reward` of the new child.
5. Backpropagate the reward from the new child through its parents, through to the root.
