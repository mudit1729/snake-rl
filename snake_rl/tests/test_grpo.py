"""Targeted regression tests for the GRPO implementation."""

import copy

import numpy as np
import pytest

try:
    import torch  # noqa: F401
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import gymnasium  # noqa: F401
    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False

from snake_rl.algos.grpo import GRPOAgent
from snake_rl.env.snake_env import SnakeEnv


@pytest.mark.skipif(
    not TORCH_AVAILABLE or not GYMNASIUM_AVAILABLE,
    reason="GRPO tests require torch and gymnasium",
)
class TestGRPOAgent:
    """Regression tests covering GRPO rollout correctness."""

    def _make_env_and_agent(self) -> tuple[SnakeEnv, GRPOAgent]:
        env = SnakeEnv(
            grid_size=(6, 6),
            observation_type="rgb",
            observation_size=(42, 42),
        )
        agent = GRPOAgent(
            input_channels=3,
            input_size=(42, 42),
            num_actions=env.action_space.n,
            group_size=8,
            device="cpu",
        )
        return env, agent

    def test_collect_group_rollouts_restores_env_state(self):
        """Lookahead should not mutate the real environment state."""
        env, agent = self._make_env_and_agent()
        obs, _ = env.reset(seed=123)

        state_before = copy.deepcopy(env.engine._state)
        episode_steps_before = env._episode_steps
        frame_buffer_before = copy.deepcopy(env._frame_buffer)

        group_actions, group_values = agent.collect_group_rollouts(env, obs)

        assert group_actions.tolist() == [0, 1, 2, 3]
        assert group_values.shape == (4,)
        assert np.isfinite(group_values).all()

        assert env._state is env.engine._state
        assert env._episode_steps == episode_steps_before
        assert env.engine._state.snake_positions == state_before.snake_positions
        assert env.engine._state.food_position == state_before.food_position
        assert env.engine._state.direction == state_before.direction
        assert env.engine._state.score == state_before.score
        assert env.engine._state.steps == state_before.steps
        assert np.array_equal(env.engine._state.grid, state_before.grid)

        assert len(env._frame_buffer) == len(frame_buffer_before)
        for before, after in zip(frame_buffer_before, env._frame_buffer):
            assert np.array_equal(after, before)

    def test_collect_trajectory_computes_returns_and_updates(self):
        """Collected trajectories should carry finite returns for value loss."""
        env, agent = self._make_env_and_agent()

        trajectory = agent.collect_trajectory(env, max_steps=30)

        assert trajectory
        assert all(np.isfinite(step.advantage) for step in trajectory)
        assert all(np.isfinite(step.return_) for step in trajectory)

        if trajectory[-1].done:
            assert trajectory[-1].return_ == pytest.approx(
                trajectory[-1].reward,
                abs=1e-4,
            )

        metrics = agent.update(
            [trajectory],
            epochs=1,
            mini_batch_size=min(8, len(trajectory)),
        )

        assert metrics["num_updates"] >= 1
        assert np.isfinite(metrics["loss"])
        assert np.isfinite(metrics["policy_loss"])
        assert np.isfinite(metrics["value_loss"])
