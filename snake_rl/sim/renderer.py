"""
Snake Game Renderer

Provides both RGB rendering (for human viewing) and array rendering (for ML training).
Optimized for performance with minimal memory allocations.
"""

from typing import Optional, Tuple
import numpy as np

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from .engine import SnakeState, CellType


class SnakeRenderer:
    """
    Renderer for snake game with support for RGB and array outputs.
    
    Features:
    - RGB rendering via pygame for human viewing
    - Fast array rendering for ML training
    - Configurable colors and scaling
    - Minimal memory allocation for performance
    """
    
    def __init__(self, cell_size: int = 20):
        """
        Initialize renderer.
        
        Args:
            cell_size: Size of each grid cell in pixels for RGB rendering
        """
        self.cell_size = cell_size
        
        # Color scheme for RGB rendering
        self.colors = {
            CellType.EMPTY: (0, 0, 0),        # Black
            CellType.SNAKE_HEAD: (0, 255, 0), # Bright green
            CellType.SNAKE_BODY: (0, 180, 0), # Dark green
            CellType.FOOD: (255, 0, 0),       # Red
            CellType.WALL: (128, 128, 128),   # Gray
        }
        
        # Initialize pygame if available
        self._pygame_initialized = False
        if PYGAME_AVAILABLE:
            self._init_pygame()
    
    def _init_pygame(self) -> None:
        """Initialize pygame for RGB rendering."""
        if not self._pygame_initialized:
            pygame.init()
            self._pygame_initialized = True
    
    def render_rgb(
        self, 
        state: SnakeState, 
        scale: int = 1
    ) -> np.ndarray:
        """
        Render state as RGB image using pygame.
        
        Args:
            state: Current game state
            scale: Scaling factor for output image (1 = original size)
            
        Returns:
            RGB image as numpy array with shape (H, W, 3)
        """
        if not PYGAME_AVAILABLE:
            raise RuntimeError("pygame not available for RGB rendering")
        
        height, width = state.grid.shape
        surface_width = width * self.cell_size
        surface_height = height * self.cell_size
        
        # Create pygame surface
        surface = pygame.Surface((surface_width, surface_height))
        surface.fill(self.colors[CellType.EMPTY])
        
        # Draw grid cells
        for row in range(height):
            for col in range(width):
                cell_type = CellType(state.grid[row, col])
                if cell_type != CellType.EMPTY:
                    color = self.colors[cell_type]
                    rect = pygame.Rect(
                        col * self.cell_size,
                        row * self.cell_size,
                        self.cell_size,
                        self.cell_size
                    )
                    pygame.draw.rect(surface, color, rect)
        
        # Convert to numpy array
        rgb_array = pygame.surfarray.array3d(surface)
        rgb_array = np.transpose(rgb_array, (1, 0, 2))  # Correct orientation
        
        # Apply scaling if needed
        if scale != 1:
            from PIL import Image
            pil_image = Image.fromarray(rgb_array)
            new_size = (surface_width * scale, surface_height * scale)
            pil_image = pil_image.resize(new_size, Image.NEAREST)
            rgb_array = np.array(pil_image)
        
        return rgb_array.astype(np.uint8)
    
    def render_array(self, state: SnakeState) -> np.ndarray:
        """
        Render state as integer array for fast ML processing.
        
        Args:
            state: Current game state
            
        Returns:
            Integer array with shape (grid_height, grid_width)
            Values correspond to CellType enum values
        """
        return state.grid.astype(np.int32)
    
    def render_observation(
        self, 
        state: SnakeState, 
        output_size: Tuple[int, int] = (84, 84),
        channels: str = "rgb"
    ) -> np.ndarray:
        """
        Render state as ML observation with specified format.
        
        Args:
            state: Current game state
            output_size: (height, width) of output observation
            channels: "rgb" for 3-channel RGB, "gray" for 1-channel grayscale
            
        Returns:
            Observation array with shape (H, W, C) where C depends on channels
        """
        if channels == "rgb":
            # Render as RGB and resize
            rgb_image = self.render_rgb(state, scale=1)
            return self._resize_observation(rgb_image, output_size)
        
        elif channels == "gray":
            # Convert grid to grayscale values
            gray_values = {
                CellType.EMPTY: 0,
                CellType.SNAKE_HEAD: 255,
                CellType.SNAKE_BODY: 128,
                CellType.FOOD: 192,
                CellType.WALL: 64,
            }
            
            height, width = state.grid.shape
            gray_image = np.zeros((height, width), dtype=np.uint8)
            
            for row in range(height):
                for col in range(width):
                    cell_type = CellType(state.grid[row, col])
                    gray_image[row, col] = gray_values[cell_type]
            
            # Resize and add channel dimension
            gray_image = self._resize_observation(gray_image, output_size)
            return np.expand_dims(gray_image, axis=-1)
        
        else:
            raise ValueError(f"Unknown channels format: {channels}")
    
    def _resize_observation(
        self, 
        image: np.ndarray, 
        output_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Resize image to target size using nearest neighbor interpolation.
        
        Args:
            image: Input image array
            output_size: Target (height, width)
            
        Returns:
            Resized image array
        """
        try:
            from PIL import Image
            
            # Handle both grayscale and RGB images
            if len(image.shape) == 2:
                pil_image = Image.fromarray(image, mode='L')
            else:
                pil_image = Image.fromarray(image, mode='RGB')
            
            # Resize with nearest neighbor to preserve pixel art style
            pil_image = pil_image.resize(
                (output_size[1], output_size[0]), 
                Image.NEAREST
            )
            
            return np.array(pil_image)
            
        except ImportError:
            # Fallback to simple nearest neighbor without PIL
            return self._simple_resize(image, output_size)
    
    def _simple_resize(
        self, 
        image: np.ndarray, 
        output_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Simple nearest neighbor resize without external dependencies.
        
        Args:
            image: Input image array
            output_size: Target (height, width)
            
        Returns:
            Resized image array
        """
        old_height, old_width = image.shape[:2]
        new_height, new_width = output_size
        
        # Calculate scaling factors
        row_scale = old_height / new_height
        col_scale = old_width / new_width
        
        # Create output array
        if len(image.shape) == 3:
            resized = np.zeros((new_height, new_width, image.shape[2]), dtype=image.dtype)
        else:
            resized = np.zeros((new_height, new_width), dtype=image.dtype)
        
        # Fill with nearest neighbor values
        for new_row in range(new_height):
            for new_col in range(new_width):
                old_row = int(new_row * row_scale)
                old_col = int(new_col * col_scale)
                
                # Clamp to valid indices
                old_row = min(old_row, old_height - 1)
                old_col = min(old_col, old_width - 1)
                
                resized[new_row, new_col] = image[old_row, old_col]
        
        return resized
    
    @staticmethod
    def create_display_surface(width: int, height: int) -> Optional[object]:
        """
        Create pygame display surface for interactive viewing.
        
        Args:
            width: Display width in pixels
            height: Display height in pixels
            
        Returns:
            Pygame display surface or None if pygame unavailable
        """
        if not PYGAME_AVAILABLE:
            return None
        
        pygame.init()
        return pygame.display.set_mode((width, height))
    
    def get_rgb_array_from_surface(self, surface) -> np.ndarray:
        """
        Convert pygame surface to RGB numpy array.
        
        Args:
            surface: Pygame surface
            
        Returns:
            RGB array with shape (H, W, 3)
        """
        if not PYGAME_AVAILABLE:
            raise RuntimeError("pygame not available")
        
        rgb_array = pygame.surfarray.array3d(surface)
        return np.transpose(rgb_array, (1, 0, 2)).astype(np.uint8)