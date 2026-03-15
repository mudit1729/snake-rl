#!/usr/bin/env python3
"""
Snake RL Demo Script

Interactive demo that allows switching between human and AI control.
Press 'H' for human control, 'A' for AI control, 'Q' to quit.
"""

import argparse
import time
from pathlib import Path

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from snake_rl.env.snake_env import SnakeEnv
from snake_rl.algos.dqn import DQNAgent
from snake_rl.sim.engine import Action


class SnakeDemo:
    """Interactive Snake demo with AI and human modes."""
    
    def __init__(
        self,
        checkpoint_path: str = None,
        grid_size: tuple = (10, 10),
        cell_size: int = 30,
        fps: int = 10
    ):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.fps = fps
        
        # Create environment
        self.env = SnakeEnv(
            grid_size=grid_size,
            observation_type="rgb",
            observation_size=(84, 84),
            render_mode="human"
        )
        
        # Load AI agent if checkpoint provided
        self.agent = None
        if checkpoint_path and TORCH_AVAILABLE:
            try:
                self.agent = self._load_agent(checkpoint_path)
                print(f"✅ AI agent loaded from: {checkpoint_path}")
            except Exception as e:
                print(f"❌ Failed to load AI agent: {e}")
        
        # Initialize pygame if available
        if PYGAME_AVAILABLE:
            pygame.init()
            pygame.display.set_caption("Snake RL Demo - Press H/A to switch modes, Q to quit")
            self.clock = pygame.time.Clock()
            
            # Calculate window size
            window_width = grid_size[1] * cell_size
            window_height = grid_size[0] * cell_size + 60  # Extra space for UI
            self.screen = pygame.display.set_mode((window_width, window_height))
            self.font = pygame.font.Font(None, 36)
        else:
            print("⚠️  Pygame not available. Running in text mode.")
            self.screen = None
    
    def _load_agent(self, checkpoint_path: str) -> DQNAgent:
        """Load trained agent from checkpoint."""
        obs_shape = self.env.observation_space.shape
        if len(obs_shape) == 3:
            input_channels = obs_shape[2]
            input_size = obs_shape[:2]
        else:
            input_channels = 1
            input_size = obs_shape
        
        agent = DQNAgent(
            input_channels=input_channels,
            input_size=input_size,
            num_actions=self.env.action_space.n,
            device="cpu"  # Use CPU for demo
        )
        
        agent.load(checkpoint_path)
        agent.set_eval_mode()
        return agent
    
    def _get_human_action(self) -> int:
        """Get action from human input."""
        if not PYGAME_AVAILABLE:
            # Text mode input
            print("Enter action (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT): ", end="")
            try:
                return int(input())
            except (ValueError, KeyboardInterrupt):
                return 0
        
        # Pygame input
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            return Action.UP
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            return Action.DOWN
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            return Action.LEFT
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            return Action.RIGHT
        
        return None  # No action
    
    def _draw_ui(self, mode: str, score: int, steps: int):
        """Draw UI elements."""
        if not PYGAME_AVAILABLE:
            return
        
        # Clear UI area
        ui_rect = pygame.Rect(0, self.grid_size[0] * self.cell_size, 
                             self.grid_size[1] * self.cell_size, 60)
        pygame.draw.rect(self.screen, (0, 0, 0), ui_rect)
        
        # Draw text
        mode_text = self.font.render(f"Mode: {mode}", True, (255, 255, 255))
        score_text = self.font.render(f"Score: {score}", True, (255, 255, 255))
        steps_text = self.font.render(f"Steps: {steps}", True, (255, 255, 255))
        help_text = self.font.render("H=Human, A=AI, Q=Quit", True, (200, 200, 200))
        
        self.screen.blit(mode_text, (10, self.grid_size[0] * self.cell_size + 5))
        self.screen.blit(score_text, (150, self.grid_size[0] * self.cell_size + 5))
        self.screen.blit(steps_text, (250, self.grid_size[0] * self.cell_size + 5))
        self.screen.blit(help_text, (10, self.grid_size[0] * self.cell_size + 35))
    
    def run(self):
        """Run the interactive demo."""
        print("🐍 Snake RL Demo")
        print("Controls:")
        print("  H - Switch to Human mode")
        print("  A - Switch to AI mode (if available)")
        print("  Q - Quit")
        if PYGAME_AVAILABLE:
            print("  Arrow keys or WASD - Move snake (in human mode)")
        print()
        
        # Initial state
        mode = "Human" if not self.agent else "AI"
        obs, _ = self.env.reset()
        last_action = Action.RIGHT
        action_timer = 0
        
        running = True
        while running:
            if PYGAME_AVAILABLE:
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            running = False
                        elif event.key == pygame.K_h:
                            mode = "Human"
                            print("Switched to Human mode")
                        elif event.key == pygame.K_a and self.agent:
                            mode = "AI"
                            print("Switched to AI mode")
            
            # Get action based on mode
            action = None
            
            if mode == "Human":
                action = self._get_human_action()
                if action is None:
                    action = last_action  # Continue in same direction
            elif mode == "AI" and self.agent:
                action = self.agent.act(obs, training=False)
            else:
                action = last_action  # Fallback
            
            # Execute action
            obs, reward, terminated, truncated, info = self.env.step(action)
            last_action = action
            
            # Render environment
            if PYGAME_AVAILABLE:
                self.env.render()
                self._draw_ui(mode, info.get("score", 0), info.get("steps", 0))
                pygame.display.flip()
                self.clock.tick(self.fps)
            else:
                # Text mode rendering
                print(f"Score: {info.get('score', 0)}, Steps: {info.get('steps', 0)}")
                time.sleep(1.0 / self.fps)
            
            # Handle episode end
            if terminated or truncated:
                print(f"Episode ended: {info.get('reason', 'unknown')}")
                print(f"Final score: {info.get('score', 0)}")
                
                if PYGAME_AVAILABLE:
                    # Show game over screen
                    game_over_text = self.font.render("Game Over! Press R to restart", True, (255, 0, 0))
                    text_rect = game_over_text.get_rect(center=(
                        self.grid_size[1] * self.cell_size // 2,
                        self.grid_size[0] * self.cell_size // 2
                    ))
                    self.screen.blit(game_over_text, text_rect)
                    pygame.display.flip()
                    
                    # Wait for restart
                    waiting = True
                    while waiting:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                running = False
                                waiting = False
                            elif event.type == pygame.KEYDOWN:
                                if event.key == pygame.K_r:
                                    obs, _ = self.env.reset()
                                    waiting = False
                                elif event.key == pygame.K_q:
                                    running = False
                                    waiting = False
                else:
                    # Text mode restart
                    print("Press Enter to restart or 'q' to quit: ", end="")
                    try:
                        response = input().strip().lower()
                        if response == 'q':
                            running = False
                        else:
                            obs, _ = self.env.reset()
                    except KeyboardInterrupt:
                        running = False
        
        if PYGAME_AVAILABLE:
            pygame.quit()
        
        print("👋 Demo ended")


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Snake RL Interactive Demo")
    parser.add_argument("--model", help="Path to trained model checkpoint")
    parser.add_argument("--grid-size", nargs=2, type=int, default=[10, 10], help="Grid size")
    parser.add_argument("--cell-size", type=int, default=30, help="Cell size in pixels")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second")
    
    args = parser.parse_args()
    
    # Check dependencies
    if not PYGAME_AVAILABLE:
        print("⚠️  Pygame not available. Running in limited text mode.")
        print("   Install pygame for full interactive experience: pip install pygame")
    
    if args.model and not TORCH_AVAILABLE:
        print("❌ PyTorch not available. Cannot load AI model.")
        print("   Install PyTorch to use AI agents: pip install torch")
        args.model = None
    
    # Create and run demo
    demo = SnakeDemo(
        checkpoint_path=args.model,
        grid_size=tuple(args.grid_size),
        cell_size=args.cell_size,
        fps=args.fps
    )
    
    try:
        demo.run()
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted")


if __name__ == "__main__":
    main()