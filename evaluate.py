#!/usr/bin/env python3
"""
Evaluation Script for Snake RL Agents

Load trained models and evaluate their performance.
Supports visualization and detailed performance analysis.
"""

import argparse
import time
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import hydra
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False

from snake_rl.env.snake_env import SnakeEnv
from snake_rl.algos.dqn import DQNAgent


def load_agent_from_checkpoint(
    checkpoint_path: str,
    env: SnakeEnv,
    device: str = "auto"
) -> DQNAgent:
    """Load agent from checkpoint file."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for model loading")
    
    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
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
    
    # Create agent with default parameters (will be overridden by checkpoint)
    agent = DQNAgent(
        input_channels=input_channels,
        input_size=input_size,
        num_actions=num_actions,
        device=device
    )
    
    # Load checkpoint
    agent.load(checkpoint_path)
    agent.set_eval_mode()
    
    return agent


def evaluate_agent_detailed(
    agent: DQNAgent,
    env: SnakeEnv,
    num_episodes: int = 100,
    max_steps: int = 1000,
    render: bool = False,
    save_replays: bool = False,
    replay_dir: str = "replays"
) -> Dict[str, Any]:
    """
    Detailed agent evaluation with statistics and optional rendering.
    
    Args:
        agent: Trained agent to evaluate
        env: Environment to evaluate on
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        render: Whether to render episodes
        save_replays: Whether to save replay data
        replay_dir: Directory to save replays
        
    Returns:
        Dictionary of evaluation metrics and statistics
    """
    episode_rewards = []
    episode_lengths = []
    episode_scores = []
    episode_reasons = []
    
    if save_replays:
        replay_path = Path(replay_dir)
        replay_path.mkdir(parents=True, exist_ok=True)
    
    print(f"🎯 Evaluating agent for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_data = [] if save_replays else None
        
        if render:
            print(f"\n🎮 Episode {episode + 1}")
            env.render()
            time.sleep(0.1)
        
        for step in range(max_steps):
            # Get action from agent
            action = agent.act(obs, training=False)
            
            # Save step data for replay
            if save_replays:
                episode_data.append({
                    "step": step,
                    "observation": obs.tolist() if isinstance(obs, np.ndarray) else obs,
                    "action": int(action)
                })
            
            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if render:
                env.render()
                print(f"Step {step + 1}: Action={action}, Reward={reward:.3f}, Score={info.get('score', 0)}")
                time.sleep(0.1)
            
            if terminated or truncated:
                reason = info.get("reason", "unknown")
                episode_reasons.append(reason)
                
                if "score" in info:
                    episode_scores.append(info["score"])
                
                if render:
                    print(f"Episode ended: {reason}, Final score: {info.get('score', 0)}")
                
                # Save replay data
                if save_replays:
                    replay_file = replay_path / f"episode_{episode:03d}.json"
                    replay_data = {
                        "episode": episode,
                        "reward": episode_reward,
                        "score": info.get("score", 0),
                        "length": episode_length,
                        "reason": reason,
                        "steps": episode_data
                    }
                    with open(replay_file, 'w') as f:
                        json.dump(replay_data, f, indent=2)
                
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if (episode + 1) % 10 == 0:
            print(f"Completed {episode + 1}/{num_episodes} episodes")
    
    # Compute statistics
    rewards_array = np.array(episode_rewards)
    lengths_array = np.array(episode_lengths)
    scores_array = np.array(episode_scores) if episode_scores else np.array([0])
    
    # Termination reason statistics
    reason_counts = {}
    for reason in episode_reasons:
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    
    statistics = {
        "num_episodes": num_episodes,
        "reward_stats": {
            "mean": float(np.mean(rewards_array)),
            "std": float(np.std(rewards_array)),
            "min": float(np.min(rewards_array)),
            "max": float(np.max(rewards_array)),
            "median": float(np.median(rewards_array)),
            "percentile_25": float(np.percentile(rewards_array, 25)),
            "percentile_75": float(np.percentile(rewards_array, 75)),
        },
        "length_stats": {
            "mean": float(np.mean(lengths_array)),
            "std": float(np.std(lengths_array)),
            "min": int(np.min(lengths_array)),
            "max": int(np.max(lengths_array)),
            "median": float(np.median(lengths_array)),
        },
        "score_stats": {
            "mean": float(np.mean(scores_array)),
            "std": float(np.std(scores_array)),
            "min": int(np.min(scores_array)),
            "max": int(np.max(scores_array)),
            "median": float(np.median(scores_array)),
            "total_food_eaten": int(np.sum(scores_array)),
        },
        "termination_reasons": reason_counts,
        "success_rate": reason_counts.get("timeout", 0) / num_episodes,
        "average_fps": num_episodes * np.mean(lengths_array) / (num_episodes * 0.01)  # Estimate
    }
    
    return statistics


def benchmark_performance(
    agent: DQNAgent,
    env: SnakeEnv,
    num_steps: int = 10000
) -> Dict[str, float]:
    """Benchmark agent inference performance."""
    print(f"🚀 Benchmarking performance for {num_steps} steps...")
    
    obs, _ = env.reset()
    
    # Warmup
    for _ in range(100):
        action = agent.act(obs, training=False)
        obs, _, done, trunc, _ = env.step(action)
        if done or trunc:
            obs, _ = env.reset()
    
    # Benchmark
    start_time = time.time()
    
    for step in range(num_steps):
        action = agent.act(obs, training=False)
        obs, _, done, trunc, _ = env.step(action)
        
        if done or trunc:
            obs, _ = env.reset()
    
    elapsed = time.time() - start_time
    steps_per_sec = num_steps / elapsed
    
    return {
        "steps_per_second": steps_per_sec,
        "ms_per_step": elapsed * 1000 / num_steps,
        "total_time": elapsed,
    }


def compare_agents(
    checkpoint_paths: List[str],
    env_config: Dict[str, Any],
    num_episodes: int = 50
) -> Dict[str, Dict[str, Any]]:
    """Compare multiple agents on the same environment."""
    results = {}
    
    for i, checkpoint_path in enumerate(checkpoint_paths):
        print(f"\n🔄 Evaluating agent {i+1}/{len(checkpoint_paths)}: {checkpoint_path}")
        
        # Create fresh environment
        env = SnakeEnv(**env_config)
        
        try:
            # Load agent
            agent = load_agent_from_checkpoint(checkpoint_path, env)
            
            # Evaluate
            stats = evaluate_agent_detailed(agent, env, num_episodes=num_episodes)
            
            # Benchmark
            perf = benchmark_performance(agent, env, num_steps=1000)
            
            results[Path(checkpoint_path).name] = {
                "evaluation": stats,
                "performance": perf,
                "checkpoint_path": checkpoint_path
            }
            
        except Exception as e:
            print(f"❌ Error evaluating {checkpoint_path}: {e}")
            results[Path(checkpoint_path).name] = {"error": str(e)}
    
    return results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate Snake RL agents")
    parser.add_argument("checkpoint", help="Path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--render", action="store_true", help="Render episodes")
    parser.add_argument("--save-replays", action="store_true", help="Save replay data")
    parser.add_argument("--replay-dir", default="replays", help="Replay directory")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--grid-size", nargs=2, type=int, default=[10, 10], help="Grid size")
    parser.add_argument("--observation-type", default="rgb", help="Observation type")
    parser.add_argument("--observation-size", nargs=2, type=int, default=[84, 84], help="Observation size")
    parser.add_argument("--reward-scheme", default="classic", help="Reward scheme")
    parser.add_argument("--output", help="Save results to JSON file")
    
    args = parser.parse_args()
    
    if not TORCH_AVAILABLE:
        print("❌ PyTorch not available. Install PyTorch to evaluate models.")
        return
    
    # Create environment
    env_config = {
        "grid_size": tuple(args.grid_size),
        "observation_type": args.observation_type,
        "observation_size": tuple(args.observation_size),
        "reward_scheme": args.reward_scheme,
        "render_mode": "human" if args.render else None,
    }
    
    env = SnakeEnv(**env_config)
    
    print(f"🐍 Snake RL Agent Evaluation")
    print(f"Environment: {env.grid_size} grid, {env.observation_space.shape} obs")
    print(f"Checkpoint: {args.checkpoint}")
    
    # Load agent
    try:
        agent = load_agent_from_checkpoint(args.checkpoint, env)
        print(f"Agent: {agent.q_network.get_parameter_count():,} parameters")
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return
    
    # Evaluate agent
    results = {}
    
    print(f"\n📊 Running detailed evaluation...")
    eval_stats = evaluate_agent_detailed(
        agent, env,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        render=args.render,
        save_replays=args.save_replays,
        replay_dir=args.replay_dir
    )
    results["evaluation"] = eval_stats
    
    # Performance benchmark
    if args.benchmark:
        print(f"\n🚀 Running performance benchmark...")
        perf_stats = benchmark_performance(agent, env)
        results["performance"] = perf_stats
        
        print(f"Performance: {perf_stats['steps_per_second']:.1f} steps/sec")
    
    # Print summary
    print(f"\n📈 Evaluation Summary:")
    print(f"   Episodes: {eval_stats['num_episodes']}")
    print(f"   Mean reward: {eval_stats['reward_stats']['mean']:.3f} ± {eval_stats['reward_stats']['std']:.3f}")
    print(f"   Mean score: {eval_stats['score_stats']['mean']:.1f} ± {eval_stats['score_stats']['std']:.1f}")
    print(f"   Max score: {eval_stats['score_stats']['max']}")
    print(f"   Mean episode length: {eval_stats['length_stats']['mean']:.1f}")
    print(f"   Success rate: {eval_stats['success_rate']:.1%}")
    
    print(f"   Termination reasons:")
    for reason, count in eval_stats['termination_reasons'].items():
        percentage = count / eval_stats['num_episodes'] * 100
        print(f"     {reason}: {count} ({percentage:.1f}%)")
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to: {args.output}")


if __name__ == "__main__":
    main()