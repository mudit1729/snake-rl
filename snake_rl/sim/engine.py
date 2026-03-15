"""
Core Snake Game Engine

High-performance, deterministic snake game implementation for RL training.
Supports both 'solid' and 'wrap' wall modes for different game variants.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class Action(IntEnum):
    """Snake movement actions."""
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class CellType(IntEnum):
    """Grid cell types."""
    EMPTY = 0
    SNAKE_HEAD = 1
    SNAKE_BODY = 2
    FOOD = 3
    WALL = 4


@dataclass
class SnakeState:
    """Complete state representation of the snake game."""
    grid: np.ndarray  # 2D grid with cell types
    snake_positions: List[Tuple[int, int]]  # [(row, col), ...] head first
    food_position: Tuple[int, int]  # (row, col)
    direction: Action
    score: int
    steps: int
    game_over: bool


class SnakeEngine:
    """
    High-performance snake game engine for RL training.
    
    Features:
    - Deterministic behavior with seeded RNG
    - Fast collision detection and movement
    - Support for solid walls and wrap-around modes
    - Optimized for >5K FPS headless simulation
    """
    
    def __init__(
        self,
        grid_size: Tuple[int, int] = (10, 10),
        wall_mode: str = "solid",
        max_steps: Optional[int] = None,
    ):
        """
        Initialize snake engine.
        
        Args:
            grid_size: (height, width) of the game grid
            wall_mode: 'solid' for wall collisions, 'wrap' for wraparound
            max_steps: Maximum steps before game over (None for no limit)
        """
        self.height, self.width = grid_size
        self.wall_mode = wall_mode
        self.max_steps = max_steps or (self.height * self.width * 4)
        
        # Validate parameters
        if self.height < 3 or self.width < 3:
            raise ValueError("Grid size must be at least 3x3")
        if wall_mode not in ("solid", "wrap"):
            raise ValueError("wall_mode must be 'solid' or 'wrap'")
        
        # Internal state
        self._rng = np.random.RandomState()
        self._state: Optional[SnakeState] = None
        self._action_to_direction = {
            Action.UP: (-1, 0),
            Action.DOWN: (1, 0),
            Action.LEFT: (0, -1),
            Action.RIGHT: (0, 1),
        }
    
    def reset(self, seed: Optional[int] = None) -> SnakeState:
        """
        Reset the game to initial state.
        
        Args:
            seed: Random seed for deterministic behavior
            
        Returns:
            Initial game state
        """
        if seed is not None:
            self._rng.seed(seed)
        
        # Initialize empty grid
        grid = np.zeros((self.height, self.width), dtype=np.int8)
        
        # Place snake in center
        center_row, center_col = self.height // 2, self.width // 2
        snake_positions = [(center_row, center_col)]
        
        # Set initial grid state
        grid[center_row, center_col] = CellType.SNAKE_HEAD
        
        # Place initial food
        food_position = self._spawn_food(grid, snake_positions)
        grid[food_position] = CellType.FOOD
        
        # Create initial state
        self._state = SnakeState(
            grid=grid,
            snake_positions=snake_positions,
            food_position=food_position,
            direction=Action.RIGHT,  # Default initial direction
            score=0,
            steps=0,
            game_over=False,
        )
        
        return self._state
    
    def step(self, action: int) -> Tuple[SnakeState, float, bool, Dict[str, Any]]:
        """
        Execute one game step.
        
        Args:
            action: Action to take (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
            
        Returns:
            (state, reward, done, info) tuple
        """
        if self._state is None:
            raise RuntimeError("Must call reset() before step()")
        
        if self._state.game_over:
            return self._state, 0.0, True, {"reason": "already_done"}
        
        action = Action(action)
        
        # Prevent immediate 180-degree turns (snake can't reverse into itself)
        if self._is_opposite_direction(action, self._state.direction):
            action = self._state.direction
        
        # Calculate new head position
        new_head = self._get_new_head_position(action)
        
        # Check for collisions
        collision_type = self._check_collision(new_head)
        
        reward = 0.0
        info: Dict[str, Any] = {}
        
        if collision_type == "wall" or collision_type == "self":
            # Game over
            self._state.game_over = True
            reward = -10.0
            info["reason"] = collision_type + "_collision"
        else:
            # Valid move - update snake position
            self._state.snake_positions.insert(0, new_head)
            self._state.direction = action
            self._state.steps += 1
            
            # Update grid: remove old head, add new head
            if len(self._state.snake_positions) > 1:
                old_head = self._state.snake_positions[1]
                self._state.grid[old_head] = CellType.SNAKE_BODY
            
            self._state.grid[new_head] = CellType.SNAKE_HEAD
            
            # Check if food was eaten
            if new_head == self._state.food_position:
                # Food eaten - grow snake and spawn new food
                self._state.score += 1
                reward = 10.0
                
                # Spawn new food
                self._state.food_position = self._spawn_food(
                    self._state.grid, self._state.snake_positions
                )
                self._state.grid[self._state.food_position] = CellType.FOOD
                
                info["food_eaten"] = True
            else:
                # No food - remove tail
                tail = self._state.snake_positions.pop()
                self._state.grid[tail] = CellType.EMPTY
                
                # Small negative reward to encourage efficiency
                reward = -0.01
            
            # Check for timeout
            if self._state.steps >= self.max_steps:
                self._state.game_over = True
                info["reason"] = "timeout"
        
        info["score"] = self._state.score
        info["steps"] = self._state.steps
        
        return self._state, reward, self._state.game_over, info
    
    def _get_new_head_position(self, action: Action) -> Tuple[int, int]:
        """Calculate new head position based on action."""
        current_head = self._state.snake_positions[0]
        delta_row, delta_col = self._action_to_direction[action]
        new_row = current_head[0] + delta_row
        new_col = current_head[1] + delta_col
        
        # Handle wall wraparound if enabled
        if self.wall_mode == "wrap":
            new_row = new_row % self.height
            new_col = new_col % self.width
        
        return (new_row, new_col)
    
    def _check_collision(self, position: Tuple[int, int]) -> str:
        """
        Check for collisions at given position.
        
        Returns:
            "wall", "self", or "none"
        """
        row, col = position
        
        # Wall collision (only for solid wall mode)
        if self.wall_mode == "solid":
            if row < 0 or row >= self.height or col < 0 or col >= self.width:
                return "wall"
        
        # Self collision
        if position in self._state.snake_positions:
            return "self"
        
        return "none"
    
    def _is_opposite_direction(self, new_action: Action, current_direction: Action) -> bool:
        """Check if new action is opposite to current direction."""
        opposites = {
            Action.UP: Action.DOWN,
            Action.DOWN: Action.UP,
            Action.LEFT: Action.RIGHT,
            Action.RIGHT: Action.LEFT,
        }
        return opposites[current_direction] == new_action
    
    def _spawn_food(
        self, grid: np.ndarray, snake_positions: List[Tuple[int, int]]
    ) -> Tuple[int, int]:
        """
        Spawn food at a random empty position.
        
        Args:
            grid: Current game grid
            snake_positions: Current snake body positions
            
        Returns:
            (row, col) position of new food
        """
        # Find all empty positions
        empty_positions = []
        for row in range(self.height):
            for col in range(self.width):
                if (row, col) not in snake_positions:
                    empty_positions.append((row, col))
        
        if not empty_positions:
            # This should rarely happen - return center as fallback
            return (self.height // 2, self.width // 2)
        
        # Randomly select empty position
        idx = self._rng.randint(len(empty_positions))
        return empty_positions[idx]
    
    def get_state(self) -> Optional[SnakeState]:
        """Get current game state."""
        return self._state
    
    def get_action_space_size(self) -> int:
        """Get size of action space."""
        return len(Action)
    
    def get_grid_shape(self) -> Tuple[int, int]:
        """Get grid dimensions."""
        return (self.height, self.width)