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
from .llm import openai_chat_completion
from pydantic import BaseModel
import tqdm
from .prompt_configs import (
    llama_3_8b_prompt_config,
    gpt_4o_prompt_config,
    RefineResponse,
)

ROOT_UCT_SCORE = 10_000


class MCTSNode(BaseModel):
    answer: str
    parent: MCTSNode | None = None
    children: list[MCTSNode] = []
    visits: int = 0
    Q: float = 0

    def add_child(self, child_node: MCTSNode):
        self.children.append(child_node)

    def __repr__(self):
        return f"MCTSNode(answer={self.answer}, Q={self.Q:.2f}, visits={self.visits})"


class SelectionPolicy(Enum):
    GREEDY = 1
    IMPORTANCE_SAMPLING = 2
    PAIRWISE_IMPORTANCE_SAMPLING = 3


class MCTSr(BaseModel):
    problem: str
    max_rollouts: int
    exploration_constant: float = 1.0
    max_children: int = 2
    epsilon: float = 1e-10
    reward_limit: int = 95
    excess_reward_penalty: int = 5
    selection_policy: SelectionPolicy = SelectionPolicy.IMPORTANCE_SAMPLING
    num_reward_samples: int = 3

    root: MCTSNode = MCTSNode(answer="I don't know.")

    # Logs
    critiques: list[str] = []
    refinements: list[str] = []
    rewards: list[float] = []
    selected_nodes: list[MCTSNode] = []

    def self_refine(self, node: MCTSNode) -> MCTSNode:
        raise NotImplementedError()

    def _evaluate_answer(self, node: MCTSNode) -> float:
        raise NotImplementedError()

    def self_evaluate(self, node: MCTSNode):
        """Evaluate the quality of the answer. Sample `num_samples` times and average the results."""

        rewards = []
        for _ in range(self.num_reward_samples):
            reward = self._evaluate_answer(node)

            if reward > self.reward_limit:
                reward -= self.excess_reward_penalty
            rewards.append(reward)

        avg_reward = sum(rewards) / self.num_reward_samples
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
        for _ in tqdm.tqdm(range(self.max_rollouts)):
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


class MCTSrLlama38B(MCTSr):
    def self_refine(self, node: MCTSNode) -> MCTSNode:
        critique_response = openai_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": llama_3_8b_prompt_config.critic_system_prompt,
                },
                {
                    "role": "user",
                    "content": "\n\n".join(
                        [
                            f"<problem>\n{self.problem}\n</problem>",
                            f"<current_answer>\n{node.answer}\n</current_answer>",
                        ]
                    ),
                },
            ],
            model=llama_3_8b_prompt_config.model,
            base_url=llama_3_8b_prompt_config.base_url,
            max_tokens=4000,
        )
        critique = critique_response.choices[0].message.content
        assert critique is not None
        self.critiques.append(critique)

        refined_answer_response = openai_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": llama_3_8b_prompt_config.refine_system_prompt,
                },
                {
                    "role": "user",
                    "content": "\n\n".join(
                        [
                            f"<problem>\n{self.problem}\n</problem>",
                            f"<current_answer>\n{node.answer}\n</current_answer>",
                            f"<critique>\n{critique}\n</critique>",
                        ]
                    ),
                },
            ],
            model=llama_3_8b_prompt_config.model,
            base_url=llama_3_8b_prompt_config.base_url,
            max_tokens=4000,
        )
        refined_answer = refined_answer_response.choices[0].message.content
        assert refined_answer is not None
        self.refinements.append(refined_answer)

        return MCTSNode(answer=refined_answer, parent=node)

    def _evaluate_answer(self, node: MCTSNode) -> float:
        messages = [
            {
                "role": "system",
                "content": llama_3_8b_prompt_config.evaluate_system_prompt,
            },
            {
                "role": "user",
                "content": "\n\n".join(
                    [
                        f"<problem>\n{self.problem}\n</problem>",
                        f"<answer>\n{node.answer}\n</answer>",
                    ]
                ),
            },
        ]
        for attempt in range(3):
            try:
                response = openai_chat_completion(
                    messages=messages,
                    model=llama_3_8b_prompt_config.model,
                    base_url=llama_3_8b_prompt_config.base_url,
                    max_tokens=4000,
                )
                assert response.choices[0].message.content is not None
                return int(response.choices[0].message.content)
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


class MCTSrGPT4o(MCTSr):
    def self_refine(self, node: MCTSNode) -> MCTSNode:
        critique_response = openai_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": gpt_4o_prompt_config.critic_system_prompt,
                },
                {
                    "role": "user",
                    "content": "\n\n".join(
                        [
                            f"<problem>\n{self.problem}\n</problem>",
                            f"<current_answer>\n{node.answer}\n</current_answer>",
                        ]
                    ),
                },
            ],
            model=gpt_4o_prompt_config.model,
            max_tokens=4000,
        )
        critique = critique_response.choices[0].message.content
        assert critique is not None
        self.critiques.append(critique)

        refined_answer_response = openai_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": gpt_4o_prompt_config.refine_system_prompt,
                },
                {
                    "role": "user",
                    "content": "\n\n".join(
                        [
                            f"<problem>\n{self.problem}\n</problem>",
                            f"<current_answer>\n{node.answer}\n</current_answer>",
                            f"<critique>\n{critique}\n</critique>",
                        ]
                    ),
                },
            ],
            model=gpt_4o_prompt_config.model,
            max_tokens=4000,
            response_format={"type": "json_object"},
        )
        refined_answer = RefineResponse.model_validate_json(
            refined_answer_response.choices[0].message.content
        )
        self.refinements.append(refined_answer)

        return MCTSNode(
            answer=f"# Thought {refined_answer.thought}\n\n# Answer\n{refined_answer.answer}",
            parent=node,
        )

    def _evaluate_answer(self, node: MCTSNode) -> float:
        messages = [
            {
                "role": "system",
                "content": gpt_4o_prompt_config.evaluate_system_prompt,
            },
            {
                "role": "user",
                "content": "\n\n".join(
                    [
                        f"<problem>\n{self.problem}\n</problem>",
                        f"<answer>\n{node.answer}\n</answer>",
                    ]
                ),
            },
        ]
        for attempt in range(3):
            try:
                response = openai_chat_completion(
                    messages=messages,
                    model=gpt_4o_prompt_config.model,
                    max_tokens=4000,
                )
                assert response.choices[0].message.content is not None
                return int(response.choices[0].message.content)
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


def print_tree(node: MCTSNode | None, level: int = 0):
    if node is None:
        return
    indent = " " * level * 2
    node_str = repr(node)
    for line in node_str.split("\n"):
        print(indent + line)
    for child in node.children:
        print_tree(child, level + 1)
