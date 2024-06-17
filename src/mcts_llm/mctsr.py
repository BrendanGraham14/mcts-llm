from __future__ import annotations

"""

Implements the MCTS + Self-Refine algorithm from
`Accessing GPT-4 level Mathematical Olympiad Solutions via Monte
Carlo Tree Self-refine with LLaMa-3 8B: A Technical Report`
by Zhang et. al.

"""

import random
import math
from collections import deque
from enum import Enum
from .llm import openai_response

ROOT_UCT_SCORE = 10_000


class MCTSNode:
    def __init__(self, answer: str, parent: MCTSNode | None = None):
        self.answer = answer
        self.parent = parent
        self.children: list[MCTSNode] = []
        self.visits = 0
        self.Q: float = 0

    def add_child(self, child_node: MCTSNode):
        self.children.append(child_node)

    def __repr__(self):
        return f"MCTSNode(answer={self.answer}, Q={self.Q:.2f}, visits={self.visits})"


class SelectionPolicy(Enum):
    GREEDY = 1
    IMPORTANCE_SAMPLING = 2
    PAIRWISE_IMPORTANCE_SAMPLING = 3


class MCTSr:
    def __init__(
        self,
        problem: str,
        max_rollouts: int,
        exploration_constant: float = 1.0,
        max_children: int = 2,
        epsilon: float = 1e-10,
        reward_limit: int = 95,
        excess_reward_penalty: int = 5,
        selection_policy: SelectionPolicy = SelectionPolicy.IMPORTANCE_SAMPLING,
    ):
        self.problem = problem
        self.max_rollouts = max_rollouts
        self.exploration_constant = exploration_constant
        self.root = MCTSNode("I don't know.")
        self.max_children = max_children

        self.reward_limit = reward_limit
        self.excess_reward_penalty = excess_reward_penalty
        self.selection_policy = selection_policy

        # For numerical stability
        self.epsilon = epsilon

    def self_refine(self, node: MCTSNode) -> MCTSNode:
        critique_response = openai_response(
            messages=[
                {
                    "role": "system",
                    "content": "Please provide a reflective or critical comment to improve the answer.",
                },
                {
                    "role": "user",
                    "content": f"# Problem\n{self.problem}\n\n# Current answer\n{node.answer}",
                },
            ],
            model="accounts/fireworks/models/llama-v3-8b-instruct",
            base_url="https://api.fireworks.ai/inference/v1",
            max_tokens=4000,
        )
        critique = critique_response.choices[0].message.content

        refined_answer_response = openai_response(
            messages=[
                {
                    "role": "system",
                    "content": "Please refine the answer based on the comment.",
                },
                {
                    "role": "user",
                    "content": f"# Problem\n{self.problem}\n\n# Current answer\n{node.answer}\n\n# Comment\n{critique}",
                },
            ],
            model="accounts/fireworks/models/llama-v3-8b-instruct",
            base_url="https://api.fireworks.ai/inference/v1",
            max_tokens=4000,
        )
        refined_answer = refined_answer_response.choices[0].message.content
        assert refined_answer is not None

        return MCTSNode(refined_answer, parent=node)

    def self_evaluate(self, node: MCTSNode):
        num_samples = 3
        rewards = []
        for _ in range(num_samples):
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Provide a reward score between -100 and 100 for the answer quality, using the strictest standards. "
                        "Do not give a full score above 95. Make sure the reward score is an integer. Return *ONLY* the score."
                    ),
                },
                {
                    "role": "user",
                    "content": f"# Problem\n{self.problem}\n\n# Answer\n{node.answer}",
                },
            ]
            for attempt in range(3):
                try:
                    response = openai_response(
                        messages=messages,
                        model="accounts/fireworks/models/llama-v3-8b-instruct",
                        base_url="https://api.fireworks.ai/inference/v1",
                        max_tokens=4000,
                    )
                    reward = int(response.choices[0].message.content)
                    break
                except ValueError:
                    messages.extend(
                        [
                            {
                                "role": "assistant",
                                "content": response.choices[0].message.content,
                            },
                            {
                                "role": "user",
                                "content": "Failed to parse reward as an integer.",
                            },
                        ]
                    )
                    if attempt == 2:
                        raise

            if reward > self.reward_limit:
                reward -= self.excess_reward_penalty
            rewards.append(reward)

        avg_reward = sum(rewards) / num_samples
        min_reward = min(rewards)

        # Average worst-case and average outcomes
        node.Q = (min_reward + avg_reward) / 2

    def backpropagate(self, node: MCTSNode):
        parent = node.parent
        while parent:
            best_child_Q = max(child.Q for child in parent.children)
            parent.Q = (parent.Q + best_child_Q) / 2
            parent.visits += 1
            parent = parent.parent

    def uct(self, node: MCTSNode):
        if not node.parent:
            # Using an arbitrarily high UCT score for the root node.
            # helps to prioritize breadth.
            return ROOT_UCT_SCORE

        return node.Q + self.exploration_constant * math.sqrt(
            math.log(node.parent.visits + 1) / (node.visits + self.epsilon)
        )

    def is_fully_expanded(self, node: MCTSNode):
        return len(node.children) >= self.max_children or any(
            child.Q > node.Q for child in node.children
        )

    def select_node(self):
        """Select a non-fully expanded node with the highest UCT value.

        A node is fully expanded if either:
        1. It has reached the max number of children
        2. Any of its children have a Q value greater than its own
        """
        candidates: list[MCTSNode] = []
        to_consider = deque([self.root])

        while to_consider:
            current_node = to_consider.popleft()
            if not self.is_fully_expanded(current_node):
                candidates.append(current_node)
            to_consider.extend(current_node.children)

        if not candidates:
            return self.root

        if self.selection_policy == SelectionPolicy.GREEDY:
            return max(candidates, key=self.uct)
        elif self.selection_policy == SelectionPolicy.IMPORTANCE_SAMPLING:
            # Sample, weighted by UCT score
            uct_scores = [self.uct(node) for node in candidates]
            selected_pair_idx = random.choices(
                range(len(candidates)), weights=uct_scores, k=1
            )[0]
            return candidates[selected_pair_idx]
        elif self.selection_policy == SelectionPolicy.PAIRWISE_IMPORTANCE_SAMPLING:
            # Sample, weighted by the difference in UCT scores between pairs
            uct_scores = [self.uct(node) for node in candidates]
            pairs = [
                (i, j) for i in range(len(candidates)) for j in range(len(candidates))
            ]
            pair_weights = [
                max(uct_scores[i], uct_scores[j]) - min(uct_scores[i], uct_scores[j])
                for i, j in pairs
            ]
            selected_pair_idx = random.choices(
                range(len(pairs)), weights=pair_weights, k=1
            )[0]
            selected_candidate_idx = max(
                pairs[selected_pair_idx], key=lambda x: uct_scores[x]
            )
            return candidates[selected_candidate_idx]
        else:
            raise ValueError(f"Invalid selection policy: {self.selection_policy}")

    def run(self):
        for _ in range(self.max_rollouts):
            node = self.select_node()
            child = self.self_refine(node)
            node.add_child(child)
            self.self_evaluate(child)
            self.backpropagate(child)

        return self.get_best_answer()

    def get_best_answer(self):
        from collections import deque

        to_visit = deque([self.root])
        best_node = self.root

        while to_visit:
            current_node = to_visit.popleft()
            if current_node.Q > best_node.Q:
                best_node = current_node
            to_visit.extend(current_node.children)

        return best_node.answer

    def print(self):
        print_tree(self.root)


def print_tree(node: MCTSNode | None, level: int = 0):
    if node is None:
        return
    indent = " " * level * 2
    node_str = str(node)
    for line in node_str.split("\n"):
        print(indent + line)
    for child in node.children:
        print_tree(child, level + 1)
