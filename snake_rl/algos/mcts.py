"""
Monte Carlo Tree Search planner for Snake.

The planner can wrap a learned policy/value agent and use environment
simulation to refine action selection during training or evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import copy
import math
import random

import numpy as np

from snake_rl.sim.engine import Action, SnakeEngine, SnakeState


@dataclass
class MCTSConfig:
    enabled: bool = False
    num_simulations: int = 32
    max_depth: int = 12
    rollout_policy: str = "agent"
    exploration_c: float = 1.4
    discount: float = 0.99
    action_selection: str = "visit"
    respect_agent_exploration: bool = True


@dataclass
class _SearchNode:
    state: SnakeState
    rng: np.random.RandomState
    terminal: bool
    reward_from_parent: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: Dict[int, "_SearchNode"] = field(default_factory=dict)
    unexpanded_actions: List[int] = field(default_factory=list)

    def mean_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTSPlanner:
    """Configurable MCTS planner that searches over SnakeEngine transitions."""

    def __init__(self, config: Optional[MCTSConfig] = None):
        self.config = config or MCTSConfig()

    def select_action(self, env, agent, observation: np.ndarray, training: bool = False) -> int:
        """Select an action using MCTS, falling back to the agent when disabled."""
        if not self.config.enabled:
            return agent.act(observation, training=training)

        if training and self.config.respect_agent_exploration:
            epsilon_scheduler = getattr(agent, "epsilon_scheduler", None)
            if epsilon_scheduler is not None:
                epsilon = epsilon_scheduler.get_epsilon()
                if np.random.rand() < epsilon:
                    return random.randrange(env.num_actions)

        root = self._build_root(env)
        if not root.unexpanded_actions and not root.children:
            return agent.act(observation, training=False)

        for _ in range(self.config.num_simulations):
            self._run_simulation(root, env, agent)

        if not root.children:
            return agent.act(observation, training=False)

        if self.config.action_selection == "value":
            best_action = max(
                root.children.items(),
                key=lambda item: item[1].reward_from_parent
                + self.config.discount * item[1].mean_value(),
            )[0]
        else:
            best_action = max(
                root.children.items(),
                key=lambda item: (item[1].visit_count, item[1].mean_value()),
            )[0]

        return best_action

    def _build_root(self, env) -> _SearchNode:
        state = copy.deepcopy(env.engine._state)
        rng = copy.deepcopy(env.engine._rng)
        return _SearchNode(
            state=state,
            rng=rng,
            terminal=state.game_over,
            unexpanded_actions=self._valid_actions(state.direction),
        )

    def _run_simulation(self, root: _SearchNode, env, agent) -> None:
        path: List[_SearchNode] = [root]
        rewards: List[float] = []
        node = root
        depth = 0

        while not node.terminal and depth < self.config.max_depth:
            if node.unexpanded_actions:
                action = node.unexpanded_actions.pop(0)
                child = self._expand(node, action, env)
                node.children[action] = child
                rewards.append(child.reward_from_parent)
                node = child
                path.append(node)
                depth += 1
                break

            action, child = self._select_child(node)
            rewards.append(child.reward_from_parent)
            node = child
            path.append(node)
            depth += 1

        remaining_depth = max(0, self.config.max_depth - depth)
        leaf_value = 0.0
        if not node.terminal and remaining_depth > 0:
            leaf_value = self._rollout(node.state, node.rng, remaining_depth, env, agent)

        value = leaf_value
        path[-1].visit_count += 1
        path[-1].value_sum += value

        for idx in range(len(path) - 2, -1, -1):
            value = rewards[idx] + self.config.discount * value
            path[idx].visit_count += 1
            path[idx].value_sum += value

    def _select_child(self, node: _SearchNode) -> Tuple[int, _SearchNode]:
        parent_log = math.log(max(1, node.visit_count))

        def score(item: Tuple[int, _SearchNode]) -> float:
            _, child = item
            q = child.reward_from_parent + self.config.discount * child.mean_value()
            u = self.config.exploration_c * math.sqrt(parent_log / max(1, child.visit_count))
            return q + u

        return max(node.children.items(), key=score)

    def _expand(self, node: _SearchNode, action: int, env) -> _SearchNode:
        next_state, next_rng, reward, done = self._simulate_step(
            env=env,
            state=node.state,
            rng=node.rng,
            action=action,
        )
        return _SearchNode(
            state=next_state,
            rng=next_rng,
            terminal=done,
            reward_from_parent=reward,
            unexpanded_actions=[] if done else self._valid_actions(next_state.direction),
        )

    def _rollout(self, state: SnakeState, rng: np.random.RandomState, depth: int, env, agent) -> float:
        rollout_state = copy.deepcopy(state)
        rollout_rng = copy.deepcopy(rng)
        total_reward = 0.0
        discount = 1.0

        for _ in range(depth):
            if rollout_state.game_over:
                break

            if self.config.rollout_policy == "random":
                action = random.choice(self._valid_actions(rollout_state.direction))
            else:
                obs = env._get_observation(rollout_state)
                action = agent.act(obs, training=False)

            rollout_state, rollout_rng, reward, done = self._simulate_step(
                env=env,
                state=rollout_state,
                rng=rollout_rng,
                action=action,
            )
            total_reward += discount * reward
            discount *= self.config.discount
            if done:
                break

        return total_reward

    def _simulate_step(
        self,
        env,
        state: SnakeState,
        rng: np.random.RandomState,
        action: int,
    ) -> Tuple[SnakeState, np.random.RandomState, float, bool]:
        engine = SnakeEngine(
            grid_size=env.grid_size,
            wall_mode=env.engine.wall_mode,
            max_steps=env.engine.max_steps,
        )
        engine._state = copy.deepcopy(state)
        engine._rng = copy.deepcopy(rng)
        next_state, reward, done, _ = engine.step(action)
        return copy.deepcopy(next_state), copy.deepcopy(engine._rng), reward, done

    def _valid_actions(self, direction: Action) -> List[int]:
        opposite = {
            Action.UP: Action.DOWN,
            Action.DOWN: Action.UP,
            Action.LEFT: Action.RIGHT,
            Action.RIGHT: Action.LEFT,
        }
        return [int(action) for action in Action if action != opposite[direction]]


def make_mcts_config(config_dict: Optional[Dict[str, Any]]) -> Optional[MCTSConfig]:
    if not config_dict:
        return None
    return MCTSConfig(**config_dict)
