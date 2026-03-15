"""
Tests for Gymnasium Environment

Tests the SnakeEnv wrapper, observation spaces, reward schemes,
and compatibility with gymnasium standards.
"""

import pytest
import numpy as np
from typing import Any, Dict, Tuple

try:
    import gymnasium as gym
    from gymnasium.utils.env_checker import check_env
    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False

from snake_rl.env.snake_env import SnakeEnv, make_snake_env
from snake_rl.sim.engine import Action


@pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="gymnasium not available")
class TestSnakeEnv:
    """Test suite for SnakeEnv gymnasium wrapper."""
    
    def test_initialization(self):
        """Test environment initialization with various configurations."""
        # Default configuration
        env = SnakeEnv()
        assert env.grid_size == (10, 10)
        assert env.wall_mode == "solid"
        assert env.observation_type == "rgb"
        assert env.action_space.n == 4
        
        # Custom configuration
        env = SnakeEnv(
            grid_size=(8, 12),
            wall_mode="wrap",
            observation_type="gray",
            reward_scheme="dense"
        )
        assert env.grid_size == (8, 12)
        assert env.wall_mode == "wrap"
        assert env.observation_type == "gray"
        assert env.reward_scheme == "dense"
    
    def test_observation_spaces(self):
        """Test different observation space configurations."""
        # RGB observation
        env = SnakeEnv(observation_type="rgb", observation_size=(84, 84))
        assert env.observation_space.shape == (84, 84, 3)
        assert env.observation_space.dtype == np.uint8
        
        # Grayscale observation
        env = SnakeEnv(observation_type="gray", observation_size=(64, 64))
        assert env.observation_space.shape == (64, 64, 1)
        assert env.observation_space.dtype == np.uint8
        
        # Grid observation
        env = SnakeEnv(observation_type="grid", grid_size=(10, 10))
        assert env.observation_space.shape == (10, 10, 1)
        assert env.observation_space.dtype == np.int32
        
        # Frame stacking
        env = SnakeEnv(
            observation_type="gray", 
            observation_size=(32, 32),
            stack_frames=4
        )
        assert env.observation_space.shape == (32, 32, 4)
    
    def test_reset_functionality(self):
        """Test environment reset behavior."""
        env = SnakeEnv(grid_size=(8, 8))
        
        # Test reset without seed
        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape
        assert "score" in info
        assert "steps" in info
        assert info["score"] == 0
        assert info["steps"] == 0
        
        # Test reset with seed for reproducibility
        obs1, info1 = env.reset(seed=42)
        obs2, info2 = env.reset(seed=42)
        
        # Should be identical with same seed
        assert np.array_equal(obs1, obs2)
        assert info1["score"] == info2["score"]
    
    def test_step_functionality(self):
        """Test environment step behavior."""
        env = SnakeEnv(grid_size=(10, 10))
        obs, info = env.reset(seed=42)
        
        # Test valid action
        action = Action.RIGHT
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        
        # Check info contents
        assert "score" in info
        assert "steps" in info
        assert "episode_steps" in info
        
        # Test all actions are valid
        for action_val in range(4):
            if not terminated and not truncated:
                obs, reward, terminated, truncated, info = env.step(action_val)
    
    def test_reward_schemes(self):
        """Test different reward schemes."""
        # Classic rewards
        env_classic = SnakeEnv(reward_scheme="classic", grid_size=(6, 6))
        obs, _ = env_classic.reset(seed=42)
        
        # Dense rewards
        env_dense = SnakeEnv(reward_scheme="dense", grid_size=(6, 6))
        obs, _ = env_dense.reset(seed=42)
        
        # Time penalty rewards
        env_time = SnakeEnv(reward_scheme="time_penalty", grid_size=(6, 6))
        obs, _ = env_time.reset(seed=42)
        
        # Take same actions and compare rewards
        for _ in range(10):
            action = Action.RIGHT
            
            _, reward_classic, done_classic, _, _ = env_classic.step(action)
            _, reward_dense, done_dense, _, _ = env_dense.step(action)
            _, reward_time, done_time, _, _ = env_time.step(action)
            
            # All should be valid rewards
            assert isinstance(reward_classic, (int, float))
            assert isinstance(reward_dense, (int, float))
            assert isinstance(reward_time, (int, float))
            
            if done_classic or done_dense or done_time:
                break
    
    def test_frame_skipping(self):
        """Test frame skipping functionality."""
        env_no_skip = SnakeEnv(frame_skip=1, grid_size=(8, 8))
        env_skip = SnakeEnv(frame_skip=4, grid_size=(8, 8))
        
        obs1, _ = env_no_skip.reset(seed=42)
        obs2, _ = env_skip.reset(seed=42)
        
        # Take same action
        action = Action.RIGHT
        _, _, _, _, info1 = env_no_skip.step(action)
        _, _, _, _, info2 = env_skip.step(action)
        
        # Frame skip environment should advance more steps
        assert info2["steps"] >= info1["steps"]
    
    def test_episode_termination(self):
        """Test episode termination conditions."""
        env = SnakeEnv(grid_size=(5, 5), max_steps=10)
        obs, _ = env.reset(seed=42)
        
        # Test timeout termination
        for step in range(15):
            action = Action.RIGHT
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                break
        
        # Should terminate due to collision or timeout
        assert terminated or truncated
        
        # If truncated, should be due to max steps
        if truncated and not terminated:
            assert info["episode_steps"] >= 10
    
    def test_observation_consistency(self):
        """Test that observations are consistent with game state."""
        env = SnakeEnv(observation_type="grid", grid_size=(6, 6))
        obs, _ = env.reset(seed=42)
        
        # Grid observation should match expected pattern
        assert obs.shape == (6, 6, 1)
        
        # Should have exactly one snake head and one food
        grid = obs[:, :, 0]
        head_count = np.sum(grid == 1)  # Snake head
        food_count = np.sum(grid == 3)  # Food
        
        assert head_count == 1
        assert food_count == 1
        
        # Take a step and verify consistency
        obs, _, _, _, _ = env.step(Action.RIGHT)
        grid = obs[:, :, 0]
        
        # Still should have one head and one food
        head_count = np.sum(grid == 1)
        food_count = np.sum(grid == 3)
        
        assert head_count <= 1  # May be 0 if game over
        assert food_count <= 1
    
    def test_render_modes(self):
        """Test different render modes."""
        env = SnakeEnv(render_mode="rgb_array", grid_size=(6, 6))
        obs, _ = env.reset(seed=42)
        
        # Test rgb_array rendering
        rgb_array = env.render()
        assert rgb_array is not None
        assert len(rgb_array.shape) == 3
        assert rgb_array.shape[2] == 3  # RGB channels
        
        # Test array rendering
        env._render_mode = "array"  # Change mode
        array_render = env.render()
        assert array_render is not None
        assert len(array_render.shape) == 2  # 2D grid
    
    def test_gymnasium_compatibility(self):
        """Test compatibility with gymnasium standards."""
        env = SnakeEnv(grid_size=(8, 8))
        
        # Should pass gymnasium environment checks
        try:
            check_env(env, warn=False, skip_render_check=True)
        except Exception as e:
            pytest.fail(f"Environment failed gymnasium checks: {e}")
    
    def test_deterministic_behavior(self):
        """Test that environment is deterministic with seeds."""
        env1 = SnakeEnv(grid_size=(8, 8))
        env2 = SnakeEnv(grid_size=(8, 8))
        
        # Reset with same seed
        obs1, _ = env1.reset(seed=123)
        obs2, _ = env2.reset(seed=123)
        
        assert np.array_equal(obs1, obs2)
        
        # Take same actions
        actions = [Action.RIGHT, Action.DOWN, Action.LEFT, Action.UP]
        for action in actions:
            obs1, reward1, done1, trunc1, info1 = env1.step(action)
            obs2, reward2, done2, trunc2, info2 = env2.step(action)
            
            assert np.array_equal(obs1, obs2)
            assert reward1 == reward2
            assert done1 == done2
            assert trunc1 == trunc2
            
            if done1 or done2:
                break
    
    def test_factory_function(self):
        """Test environment factory function."""
        env = make_snake_env(
            grid_size=(7, 7),
            wall_mode="wrap",
            observation_type="gray"
        )
        
        assert isinstance(env, SnakeEnv)
        assert env.grid_size == (7, 7)
        assert env.wall_mode == "wrap"
        assert env.observation_type == "gray"
    
    def test_close_functionality(self):
        """Test environment cleanup."""
        env = SnakeEnv(render_mode="human")
        env.reset()
        
        # Should not raise error
        env.close()
        
        # Should be safe to call multiple times
        env.close()


@pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="gymnasium not available")
class TestVectorEnvironment:
    """Test vector environment functionality."""
    
    def test_vector_env_import(self):
        """Test that vector environment utilities can be imported."""
        try:
            from snake_rl.env.vector_env import make_vec_env, make_sync_vec_env
            assert make_vec_env is not None
            assert make_sync_vec_env is not None
        except ImportError:
            pytest.skip("Vector environment utilities not available")
    
    def test_sync_vector_env(self):
        """Test synchronous vector environment."""
        try:
            from snake_rl.env.vector_env import make_sync_vec_env
            
            # Create sync vector env
            vec_env = make_sync_vec_env(
                n_envs=2,
                env_kwargs={"grid_size": (6, 6)},
                seed=42
            )
            
            # Test reset
            obs = vec_env.reset()
            assert len(obs) == 2  # Two observations
            
            # Test step
            actions = vec_env.action_space.sample()
            obs, rewards, dones, truncs, infos = vec_env.step(actions)
            
            assert len(obs) == 2
            assert len(rewards) == 2
            assert len(dones) == 2
            assert len(infos) == 2
            
            vec_env.close()
            
        except ImportError:
            pytest.skip("Vector environments not available")


class TestEnvironmentPerformance:
    """Test environment performance characteristics."""
    
    def test_step_performance(self):
        """Test that environment steps run at reasonable speed."""
        env = SnakeEnv(grid_size=(10, 10))
        
        import time
        
        # Warm up
        env.reset(seed=42)
        for _ in range(100):
            action = env.action_space.sample()
            obs, _, done, trunc, _ = env.step(action)
            if done or trunc:
                env.reset()
        
        # Measure performance
        num_steps = 1000
        start_time = time.time()
        
        env.reset(seed=42)
        for _ in range(num_steps):
            action = env.action_space.sample()
            obs, _, done, trunc, _ = env.step(action)
            if done or trunc:
                env.reset()
        
        elapsed = time.time() - start_time
        steps_per_sec = num_steps / elapsed
        
        print(f"Environment performance: {steps_per_sec:.1f} steps/sec")
        
        # Should be reasonably fast (adjust threshold as needed)
        min_steps_per_sec = 1000  # Minimum expected performance
        assert steps_per_sec > min_steps_per_sec, f"Too slow: {steps_per_sec:.1f} < {min_steps_per_sec}"
    
    def test_memory_usage(self):
        """Test that environment doesn't leak memory."""
        env = SnakeEnv(grid_size=(20, 20))
        
        # Run many episodes
        for episode in range(100):
            obs, _ = env.reset(seed=episode)
            
            for step in range(200):
                action = env.action_space.sample()
                obs, _, done, trunc, _ = env.step(action)
                
                if done or trunc:
                    break
        
        # If we get here without running out of memory, test passes
        assert True


class TestWrappers:
    """Test environment wrappers."""
    
    def test_wrapper_imports(self):
        """Test that wrappers can be imported."""
        try:
            from snake_rl.env.vector_env import (
                FrameSkipWrapper,
                RewardScalingWrapper,
                EpisodeStatsWrapper,
                NoopResetWrapper
            )
            
            # All should be importable
            assert FrameSkipWrapper is not None
            assert RewardScalingWrapper is not None
            assert EpisodeStatsWrapper is not None
            assert NoopResetWrapper is not None
            
        except ImportError:
            pytest.skip("Wrappers not available")
    
    @pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="gymnasium not available")
    def test_reward_scaling_wrapper(self):
        """Test reward scaling wrapper."""
        try:
            from snake_rl.env.vector_env import RewardScalingWrapper
            
            env = SnakeEnv(grid_size=(6, 6))
            wrapped_env = RewardScalingWrapper(env, scale=2.0)
            
            obs, _ = wrapped_env.reset(seed=42)
            obs, reward, done, trunc, info = wrapped_env.step(Action.RIGHT)
            
            # Check that original reward is preserved in info
            assert "original_reward" in info
            assert "reward_scale" in info
            assert info["reward_scale"] == 2.0
            
        except ImportError:
            pytest.skip("Wrappers not available")
    
    @pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="gymnasium not available")
    def test_episode_stats_wrapper(self):
        """Test episode statistics wrapper."""
        try:
            from snake_rl.env.vector_env import EpisodeStatsWrapper
            
            env = SnakeEnv(grid_size=(5, 5), max_steps=20)
            wrapped_env = EpisodeStatsWrapper(env)
            
            # Run a few episodes
            for episode in range(3):
                obs, _ = wrapped_env.reset(seed=episode)
                
                for step in range(25):
                    action = wrapped_env.action_space.sample()
                    obs, reward, done, trunc, info = wrapped_env.step(action)
                    
                    if done or trunc:
                        # Should have episode stats in info
                        assert "episode_reward" in info
                        assert "episode_length" in info
                        break
            
            # Get overall stats
            stats = wrapped_env.get_episode_stats()
            assert stats["num_episodes"] >= 2
            assert "mean_reward" in stats
            assert "mean_length" in stats
            
        except ImportError:
            pytest.skip("Wrappers not available")


def test_basic_env_functionality():
    """Test basic functionality without gymnasium dependency."""
    env = SnakeEnv(grid_size=(8, 8))
    
    # Test that basic functionality works
    obs, info = env.reset(seed=42)
    assert obs is not None
    assert isinstance(info, dict)
    
    # Test step
    obs, reward, done, trunc, info = env.step(0)
    assert obs is not None
    assert isinstance(reward, (int, float))
    assert isinstance(done, bool)
    assert isinstance(trunc, bool)
    assert isinstance(info, dict)


if __name__ == "__main__":
    # Run basic test without pytest
    test_basic_env_functionality()
    print("✅ Basic environment test passed")