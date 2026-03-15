#!/usr/bin/env python3
"""
DQN Training Script for Snake Game

Train DQN agents on Snake environment with Hydra configuration management.
Supports TensorBoard and W&B logging, checkpointing, and evaluation.
"""

import os
import time
import random
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

try:
    import hydra
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Import our modules
from snake_rl.env.snake_env import SnakeEnv
from snake_rl.algos.dqn import DQNAgent


class TrainingLogger:
    """Handles logging to multiple backends."""
    
    def __init__(
        self,
        log_dir: str,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_config: Optional[Dict[str, Any]] = None
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.tensorboard_writer = None
        if use_tensorboard and TENSORBOARD_AVAILABLE:
            self.tensorboard_writer = SummaryWriter(str(self.log_dir / "tensorboard"))
            print(f"📊 TensorBoard logging to: {self.log_dir / 'tensorboard'}")
        
        # Weights & Biases
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb_config = wandb_config or {}
            wandb.init(**wandb_config)
            print("📊 W&B logging initialized")
    
    def log_scalar(self, name: str, value: float, step: int):
        """Log scalar value to all backends."""
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar(name, value, step)
        
        if self.use_wandb:
            wandb.log({name: value}, step=step)
    
    def log_dict(self, metrics: Dict[str, float], step: int):
        """Log dictionary of metrics."""
        for name, value in metrics.items():
            self.log_scalar(name, value, step)
    
    def close(self):
        """Close all loggers."""
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        if self.use_wandb:
            wandb.finish()


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


def create_env(env_config: DictConfig) -> SnakeEnv:
    """Create environment from config."""
    # Convert config to dict and remove _target_
    env_kwargs = OmegaConf.to_container(env_config, resolve=True)
    if "_target_" in env_kwargs:
        del env_kwargs["_target_"]
    
    return SnakeEnv(**env_kwargs)


def create_agent(agent_config: DictConfig, env: SnakeEnv) -> DQNAgent:
    """Create agent from config."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for DQN training")
    
    # Get observation space info
    obs_shape = env.observation_space.shape
    if len(obs_shape) == 3:
        input_channels = obs_shape[2]
        input_size = obs_shape[:2]
    elif len(obs_shape) == 2:
        input_channels = 1
        input_size = obs_shape
    else:
        raise ValueError(f"Unsupported observation shape: {obs_shape}")
    
    num_actions = env.action_space.n
    
    # Convert config to dict and remove _target_
    agent_kwargs = OmegaConf.to_container(agent_config, resolve=True)
    if "_target_" in agent_kwargs:
        del agent_kwargs["_target_"]
    
    return DQNAgent(
        input_channels=input_channels,
        input_size=input_size,
        num_actions=num_actions,
        **agent_kwargs
    )


def evaluate_agent(
    agent: DQNAgent,
    env: SnakeEnv,
    num_episodes: int = 10,
    max_steps: int = 1000
) -> Dict[str, float]:
    """Evaluate agent performance."""
    agent.set_eval_mode()
    
    episode_rewards = []
    episode_lengths = []
    episode_scores = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            action = agent.act(obs, training=False)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if terminated or truncated:
                if "score" in info:
                    episode_scores.append(info["score"])
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    agent.set_train_mode()
    
    metrics = {
        "eval/mean_reward": np.mean(episode_rewards),
        "eval/std_reward": np.std(episode_rewards),
        "eval/mean_length": np.mean(episode_lengths),
        "eval/std_length": np.std(episode_lengths),
    }
    
    if episode_scores:
        metrics.update({
            "eval/mean_score": np.mean(episode_scores),
            "eval/max_score": np.max(episode_scores),
        })
    
    return metrics


def train_dqn(cfg: DictConfig):
    """Main training loop."""
    print("🐍 Starting Snake DQN Training")
    print(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Set random seeds
    set_random_seeds(cfg.experiment.seed)
    
    # Create directories
    checkpoint_dir = Path(cfg.checkpoints.save_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize logging
    logger = TrainingLogger(
        log_dir=cfg.logging.log_dir,
        use_tensorboard=cfg.logging.use_tensorboard,
        use_wandb=cfg.logging.use_wandb,
        wandb_config=OmegaConf.to_container(cfg.wandb, resolve=True) if cfg.logging.use_wandb else None
    )
    
    # Create environment and agent
    env = create_env(cfg.env)
    agent = create_agent(cfg.agent, env)
    
    print(f"Environment: {env.grid_size} grid, {env.observation_space.shape} obs")
    print(f"Agent: {agent.q_network.get_parameter_count():,} parameters")
    
    # Training state
    step_count = 0
    episode_count = 0
    best_eval_score = -float('inf')
    
    # Training loop
    start_time = time.time()
    
    try:
        while step_count < cfg.training.total_steps:
            # Start episode
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            episode_count += 1
            
            while step_count < cfg.training.total_steps:
                # Select and execute action
                action = agent.act(obs, training=True)
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                # Store experience
                agent.store_experience(
                    obs, action, reward, 
                    next_obs if not (terminated or truncated) else None,
                    terminated or truncated
                )
                
                # Update agent
                if len(agent.replay_buffer) >= agent.batch_size:
                    metrics = agent.update()
                    
                    # Log training metrics
                    if metrics and step_count % cfg.training.log_frequency == 0:
                        logger.log_dict(metrics, step_count)
                
                # Update state
                obs = next_obs
                episode_reward += reward
                episode_length += 1
                step_count += 1
                agent.step_count = step_count
                
                # Log episode metrics
                if terminated or truncated:
                    logger.log_scalar("train/episode_reward", episode_reward, step_count)
                    logger.log_scalar("train/episode_length", episode_length, step_count)
                    if "score" in info:
                        logger.log_scalar("train/episode_score", info["score"], step_count)
                    
                    if step_count % cfg.training.log_frequency == 0:
                        elapsed = time.time() - start_time
                        steps_per_sec = step_count / elapsed
                        print(f"Step {step_count:7d} | Episode {episode_count:5d} | "
                              f"Reward: {episode_reward:7.2f} | Score: {info.get('score', 0):3d} | "
                              f"Steps/sec: {steps_per_sec:.1f}")
                    
                    break
                
                # Evaluation
                if step_count % cfg.training.eval_frequency == 0:
                    print(f"\n🎯 Evaluating at step {step_count}...")
                    eval_metrics = evaluate_agent(agent, env, cfg.training.eval_episodes)
                    logger.log_dict(eval_metrics, step_count)
                    
                    eval_score = eval_metrics.get("eval/mean_score", eval_metrics["eval/mean_reward"])
                    print(f"Eval score: {eval_score:.2f} (best: {best_eval_score:.2f})")
                    
                    # Save best model
                    if eval_score > best_eval_score:
                        best_eval_score = eval_score
                        if cfg.checkpoints.keep_best:
                            best_path = checkpoint_dir / "best_model.pt"
                            agent.save(str(best_path))
                            print(f"💾 New best model saved: {best_path}")
                
                # Save checkpoint
                if step_count % cfg.training.save_frequency == 0:
                    checkpoint_path = checkpoint_dir / f"checkpoint_{step_count}.pt"
                    agent.save(str(checkpoint_path))
                    print(f"💾 Checkpoint saved: {checkpoint_path}")
                
                # Save last checkpoint
                if cfg.checkpoints.save_last and step_count % 10000 == 0:
                    last_path = checkpoint_dir / "last_model.pt"
                    agent.save(str(last_path))
    
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    
    finally:
        # Save final model
        final_path = checkpoint_dir / "final_model.pt"
        agent.save(str(final_path))
        print(f"💾 Final model saved: {final_path}")
        
        # Close logger
        logger.close()
        
        # Training summary
        elapsed = time.time() - start_time
        print(f"\n📊 Training Summary:")
        print(f"   Total steps: {step_count:,}")
        print(f"   Total episodes: {episode_count:,}")
        print(f"   Training time: {elapsed/3600:.2f} hours")
        print(f"   Steps per second: {step_count/elapsed:.1f}")
        print(f"   Best eval score: {best_eval_score:.2f}")


# Hydra main function
if HYDRA_AVAILABLE:
    @hydra.main(version_base=None, config_path="snake_rl/conf", config_name="config")
    def main(cfg: DictConfig) -> None:
        train_dqn(cfg)
else:
    def main():
        print("❌ Hydra not available. Install hydra-core to use this training script.")
        print("For basic training without Hydra, see examples in the repository.")


if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("❌ PyTorch not available. Install PyTorch to train DQN models.")
        exit(1)
    
    if not HYDRA_AVAILABLE:
        print("❌ Hydra not available. Install hydra-core for configuration management.")
        exit(1)
    
    main()