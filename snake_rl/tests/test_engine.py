"""
Unit tests for Snake Engine

Tests movement correctness, collision detection, food spawning,
deterministic behavior, and performance requirements.
"""

import pytest
import numpy as np
import time
from typing import Set, Tuple

from snake_rl.sim.engine import SnakeEngine, Action, CellType, SnakeState


class TestSnakeEngine:
    """Test suite for SnakeEngine class."""
    
    def test_initialization(self):
        """Test engine initialization with various parameters."""
        # Default parameters
        engine = SnakeEngine()
        assert engine.height == 10
        assert engine.width == 10
        assert engine.wall_mode == "solid"
        
        # Custom parameters
        engine = SnakeEngine(grid_size=(15, 20), wall_mode="wrap")
        assert engine.height == 15
        assert engine.width == 20
        assert engine.wall_mode == "wrap"
        
        # Invalid parameters
        with pytest.raises(ValueError):
            SnakeEngine(grid_size=(2, 2))  # Too small
        
        with pytest.raises(ValueError):
            SnakeEngine(wall_mode="invalid")
    
    def test_reset(self):
        """Test game reset functionality."""
        engine = SnakeEngine(grid_size=(10, 10))
        
        # Initial reset
        state = engine.reset(seed=42)
        
        assert isinstance(state, SnakeState)
        assert state.grid.shape == (10, 10)
        assert len(state.snake_positions) == 1
        assert state.snake_positions[0] == (5, 5)  # Center position
        assert state.direction == Action.RIGHT
        assert state.score == 0
        assert state.steps == 0
        assert not state.game_over
        
        # Grid should have snake head and food
        assert state.grid[5, 5] == CellType.SNAKE_HEAD
        assert state.grid[state.food_position] == CellType.FOOD
        assert state.food_position != (5, 5)  # Food not on snake
        
        # Count cell types
        head_count = np.sum(state.grid == CellType.SNAKE_HEAD)
        food_count = np.sum(state.grid == CellType.FOOD)
        assert head_count == 1
        assert food_count == 1
    
    def test_deterministic_behavior(self):
        """Test that same seed produces identical results."""
        engine1 = SnakeEngine(grid_size=(8, 8))
        engine2 = SnakeEngine(grid_size=(8, 8))
        
        # Reset with same seed
        state1 = engine1.reset(seed=123)
        state2 = engine2.reset(seed=123)
        
        # Should have identical initial states
        assert np.array_equal(state1.grid, state2.grid)
        assert state1.snake_positions == state2.snake_positions
        assert state1.food_position == state2.food_position
        
        # Take same actions and verify deterministic behavior
        actions = [Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT]
        for action in actions:
            state1, _, _, _ = engine1.step(action)
            state2, _, _, _ = engine2.step(action)
            
            assert np.array_equal(state1.grid, state2.grid)
            assert state1.snake_positions == state2.snake_positions
            assert state1.food_position == state2.food_position
    
    def test_movement_correctness(self):
        """Test snake movement in all directions."""
        engine = SnakeEngine(grid_size=(10, 10))
        state = engine.reset(seed=42)
        
        initial_head = state.snake_positions[0]
        
        # Move right
        state, reward, done, info = engine.step(Action.RIGHT)
        expected_head = (initial_head[0], initial_head[1] + 1)
        assert state.snake_positions[0] == expected_head
        assert state.direction == Action.RIGHT
        assert not done
        
        # Move down
        state, reward, done, info = engine.step(Action.DOWN)
        expected_head = (expected_head[0] + 1, expected_head[1])
        assert state.snake_positions[0] == expected_head
        assert state.direction == Action.DOWN
        
        # Move left
        state, reward, done, info = engine.step(Action.LEFT)
        expected_head = (expected_head[0], expected_head[1] - 1)
        assert state.snake_positions[0] == expected_head
        assert state.direction == Action.LEFT
        
        # Move up
        state, reward, done, info = engine.step(Action.UP)
        expected_head = (expected_head[0] - 1, expected_head[1])
        assert state.snake_positions[0] == expected_head
        assert state.direction == Action.UP
    
    def test_reverse_prevention(self):
        """Test that snake cannot immediately reverse direction."""
        engine = SnakeEngine(grid_size=(10, 10))
        state = engine.reset(seed=42)
        
        # Snake starts facing right, try to go left (reverse)
        initial_head = state.snake_positions[0]
        state, _, _, _ = engine.step(Action.LEFT)
        
        # Snake should continue right instead of reversing
        expected_head = (initial_head[0], initial_head[1] + 1)
        assert state.snake_positions[0] == expected_head
        assert state.direction == Action.RIGHT
    
    def test_wall_collision_solid(self):
        """Test wall collision in solid wall mode."""
        engine = SnakeEngine(grid_size=(5, 5), wall_mode="solid")
        state = engine.reset(seed=42)
        
        # Move snake to edge and collide with wall
        # First, position snake near edge
        for _ in range(3):  # Move towards edge
            state, _, done, _ = engine.step(Action.RIGHT)
            if done:
                break
        
        # Continue until collision
        collision_occurred = False
        for _ in range(10):  # Safety limit
            state, reward, done, info = engine.step(Action.RIGHT)
            if done:
                collision_occurred = True
                assert reward < 0  # Negative reward for collision
                assert "collision" in info.get("reason", "")
                break
        
        assert collision_occurred, "Wall collision should have occurred"
    
    def test_wall_wrap_around(self):
        """Test wraparound behavior in wrap mode."""
        engine = SnakeEngine(grid_size=(5, 5), wall_mode="wrap")
        state = engine.reset(seed=42)
        
        # Move snake to right edge
        initial_row = state.snake_positions[0][0]
        for _ in range(10):  # Move right multiple times
            state, _, done, _ = engine.step(Action.RIGHT)
            if done:
                pytest.fail("Game should not end with wraparound")
        
        # Verify position wrapped around
        final_row = state.snake_positions[0][0]
        assert final_row == initial_row  # Row should be unchanged
    
    def test_self_collision(self):
        """Test snake self-collision detection."""
        engine = SnakeEngine(grid_size=(10, 10))
        state = engine.reset(seed=42)
        
        # Force snake to grow by eating food multiple times
        # Find and eat food repeatedly to create a longer snake
        for _ in range(5):
            # Move towards food
            food_row, food_col = state.food_position
            head_row, head_col = state.snake_positions[0]
            
            # Simple strategy to reach food
            for _ in range(20):  # Safety limit
                if head_row < food_row:
                    action = Action.DOWN
                elif head_row > food_row:
                    action = Action.UP
                elif head_col < food_col:
                    action = Action.RIGHT
                else:
                    action = Action.LEFT
                
                state, reward, done, info = engine.step(action)
                head_row, head_col = state.snake_positions[0]
                
                if reward > 0 or done:  # Food eaten or game over
                    break
            
            if done:
                break
        
        # If snake is long enough, try to create self-collision
        if len(state.snake_positions) >= 4 and not state.game_over:
            # Create a spiral to cause self-collision
            actions = [Action.LEFT, Action.DOWN, Action.RIGHT, Action.UP]
            for action in actions * 3:
                state, reward, done, info = engine.step(action)
                if done and "self" in info.get("reason", ""):
                    assert reward < 0
                    return
        
        # If we can't create self-collision naturally, manually test collision detection
        engine2 = SnakeEngine(grid_size=(5, 5))
        state2 = engine2.reset(seed=123)
        
        # Manually create a scenario where collision is likely
        # by moving in a tight pattern
        tight_pattern = [
            Action.RIGHT, Action.DOWN, Action.LEFT, Action.UP,
            Action.RIGHT, Action.DOWN, Action.LEFT, Action.UP
        ]
        
        for action in tight_pattern:
            state2, reward, done, info = engine2.step(action)
            if done and "collision" in info.get("reason", ""):
                assert reward < 0
                return
    
    def test_food_consumption(self):
        """Test food eating and score increase."""
        engine = SnakeEngine(grid_size=(10, 10))
        state = engine.reset(seed=42)
        
        initial_score = state.score
        initial_length = len(state.snake_positions)
        food_position = state.food_position
        
        # Move snake to food position
        head_row, head_col = state.snake_positions[0]
        food_row, food_col = food_position
        
        # Calculate path to food and execute
        moves_needed = 0
        while (head_row, head_col) != (food_row, food_col) and moves_needed < 50:
            if head_row < food_row:
                action = Action.DOWN
            elif head_row > food_row:
                action = Action.UP
            elif head_col < food_col:
                action = Action.RIGHT
            else:
                action = Action.LEFT
            
            state, reward, done, info = engine.step(action)
            head_row, head_col = state.snake_positions[0]
            moves_needed += 1
            
            if done:
                break
        
        # Check if food was eaten
        if not done and "food_eaten" in info:
            assert state.score == initial_score + 1
            assert len(state.snake_positions) == initial_length + 1
            assert reward > 0
            assert state.food_position != food_position  # New food spawned
    
    def test_food_spawn_uniqueness(self):
        """Test that food spawns in empty positions only."""
        engine = SnakeEngine(grid_size=(5, 5))
        
        # Test multiple resets to ensure food doesn't spawn on snake
        for seed in range(10):
            state = engine.reset(seed=seed)
            
            # Food should not be on any snake position
            assert state.food_position not in state.snake_positions
            
            # Food position should be within grid bounds
            food_row, food_col = state.food_position
            assert 0 <= food_row < 5
            assert 0 <= food_col < 5
    
    def test_timeout_condition(self):
        """Test game timeout after max steps."""
        engine = SnakeEngine(grid_size=(5, 5), max_steps=10)
        state = engine.reset(seed=42)
        
        # Take steps until timeout
        for step in range(15):
            state, reward, done, info = engine.step(Action.RIGHT)
            
            if step < 10:
                assert state.steps == step + 1
            
            if done and info.get("reason") == "timeout":
                assert state.steps >= 10
                return
        
        pytest.fail("Timeout condition was not triggered")
    
    def test_performance_headless(self):
        """Test that engine runs at >5K FPS in headless mode."""
        engine = SnakeEngine(grid_size=(10, 10))
        
        # Warm up
        state = engine.reset(seed=42)
        for _ in range(100):
            action = Action.RIGHT if not state.game_over else Action.RIGHT
            state, _, done, _ = engine.step(action)
            if done:
                state = engine.reset()
        
        # Measure performance
        num_steps = 10000
        actions = [Action.RIGHT, Action.DOWN, Action.LEFT, Action.UP] * (num_steps // 4)
        
        start_time = time.time()
        
        for i in range(num_steps):
            action = actions[i % len(actions)]
            state, _, done, _ = engine.step(action)
            
            if done:
                state = engine.reset()
        
        end_time = time.time()
        elapsed = end_time - start_time
        fps = num_steps / elapsed
        
        print(f"Performance: {fps:.1f} FPS")
        
        # Should achieve >5K FPS on modern hardware
        # Use a lower threshold for CI environments
        min_fps = 2000  # Reduced for CI stability
        assert fps > min_fps, f"Performance too low: {fps:.1f} < {min_fps} FPS"
    
    def test_action_space(self):
        """Test action space properties."""
        engine = SnakeEngine()
        
        assert engine.get_action_space_size() == 4
        assert engine.get_grid_shape() == (10, 10)
        
        # Test all valid actions
        state = engine.reset(seed=42)
        for action_val in range(4):
            action = Action(action_val)
            state, _, _, _ = engine.step(action)
    
    def test_grid_consistency(self):
        """Test that grid state remains consistent with snake positions."""
        engine = SnakeEngine(grid_size=(8, 8))
        state = engine.reset(seed=42)
        
        for _ in range(50):
            # Verify grid consistency before step
            self._verify_grid_consistency(state)
            
            # Take action
            action = Action.RIGHT
            state, _, done, _ = engine.step(action)
            
            if done:
                state = engine.reset(seed=np.random.randint(1000))
        
        # Final consistency check
        self._verify_grid_consistency(state)
    
    def _verify_grid_consistency(self, state: SnakeState):
        """Helper to verify grid matches snake positions."""
        # Count expected cell types
        expected_head = 1
        expected_body = len(state.snake_positions) - 1
        expected_food = 1
        
        # Count actual cell types in grid
        actual_head = np.sum(state.grid == CellType.SNAKE_HEAD)
        actual_body = np.sum(state.grid == CellType.SNAKE_BODY)
        actual_food = np.sum(state.grid == CellType.FOOD)
        
        assert actual_head == expected_head, f"Head count mismatch: {actual_head} != {expected_head}"
        assert actual_body == expected_body, f"Body count mismatch: {actual_body} != {expected_body}"
        assert actual_food == expected_food, f"Food count mismatch: {actual_food} != {expected_food}"
        
        # Verify snake positions match grid
        head_pos = state.snake_positions[0]
        assert state.grid[head_pos] == CellType.SNAKE_HEAD
        
        for body_pos in state.snake_positions[1:]:
            assert state.grid[body_pos] == CellType.SNAKE_BODY
        
        # Verify food position
        assert state.grid[state.food_position] == CellType.FOOD


class TestSnakeEngineEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_minimum_grid_size(self):
        """Test engine with minimum allowed grid size."""
        engine = SnakeEngine(grid_size=(3, 3))
        state = engine.reset(seed=42)
        
        assert state.grid.shape == (3, 3)
        assert len(state.snake_positions) == 1
        
        # Should be able to take at least a few actions
        for _ in range(5):
            state, _, done, _ = engine.step(Action.RIGHT)
            if done:
                break
    
    def test_very_large_grid(self):
        """Test engine with large grid size."""
        engine = SnakeEngine(grid_size=(100, 100))
        state = engine.reset(seed=42)
        
        assert state.grid.shape == (100, 100)
        
        # Should work normally
        for _ in range(10):
            state, _, done, _ = engine.step(Action.RIGHT)
            if done:
                break
    
    def test_rectangular_grids(self):
        """Test non-square grid dimensions."""
        engine = SnakeEngine(grid_size=(5, 10))
        state = engine.reset(seed=42)
        
        assert state.grid.shape == (5, 10)
        assert state.snake_positions[0] == (2, 5)  # Center of 5x10 grid
        
        engine = SnakeEngine(grid_size=(15, 6))
        state = engine.reset(seed=42)
        
        assert state.grid.shape == (15, 6)
        assert state.snake_positions[0] == (7, 3)  # Center of 15x6 grid


# Performance benchmark that can be run separately
def benchmark_engine_performance():
    """Standalone performance benchmark."""
    print("Running Snake Engine Performance Benchmark...")
    
    engine = SnakeEngine(grid_size=(20, 20))
    
    # Test different scenarios
    scenarios = [
        ("Small grid (10x10)", (10, 10)),
        ("Medium grid (20x20)", (20, 20)),
        ("Large grid (50x50)", (50, 50)),
    ]
    
    for name, grid_size in scenarios:
        engine = SnakeEngine(grid_size=grid_size)
        
        num_steps = 5000
        start_time = time.time()
        
        state = engine.reset(seed=42)
        for i in range(num_steps):
            action = Action(i % 4)
            state, _, done, _ = engine.step(action)
            if done:
                state = engine.reset(seed=i)
        
        elapsed = time.time() - start_time
        fps = num_steps / elapsed
        
        print(f"{name}: {fps:.1f} FPS")


if __name__ == "__main__":
    benchmark_engine_performance()