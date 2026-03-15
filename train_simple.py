#!/usr/bin/env python3
"""
Simplified DQN Training Script

A minimal training script that works with basic dependencies and demonstrates
the core training loop without external logging dependencies.
Generates training visualization plots at the end.
"""

import os
import sys
import time
import random
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
    print("✅ PyTorch available")
except ImportError:
    TORCH_AVAILABLE = False
    print("❌ PyTorch not available")

try:
    import gymnasium as gym
    GYMNASIUM_AVAILABLE = True
    print("✅ Gymnasium available")
except ImportError:
    GYMNASIUM_AVAILABLE = False
    print("❌ Gymnasium not available")

# Import our modules
from snake_rl.sim.engine import SnakeEngine, Action
from snake_rl.sim.renderer import SnakeRenderer


class SimpleSnakeEnv:
    """Simplified Snake environment without Gymnasium dependency."""

    def __init__(self, grid_size=(10, 10), observation_size=(84, 84)):
        self.grid_size = grid_size
        self.observation_size = observation_size
        self.engine = SnakeEngine(grid_size=grid_size)
        self.renderer = SnakeRenderer()

        # Action space
        self.num_actions = 4

        # Observation space dimensions
        self.obs_shape = (*observation_size, 3)  # RGB

    def reset(self, seed=None):
        """Reset environment."""
        state = self.engine.reset(seed=seed)
        obs = self._get_observation(state)
        info = {"score": state.score, "steps": state.steps}
        return obs, info

    def step(self, action):
        """Execute one step."""
        state, reward, done, info = self.engine.step(action)
        obs = self._get_observation(state)
        return obs, reward, done, False, info  # terminated, truncated format

    def _get_observation(self, state):
        """Convert state to RGB observation."""
        try:
            rgb_obs = self.renderer.render_rgb(state, scale=1)
            return self._resize_observation(rgb_obs)
        except Exception:
            grid_obs = self.renderer.render_array(state)
            h, w = grid_obs.shape
            rgb_obs = np.zeros((h, w, 3), dtype=np.uint8)
            rgb_obs[grid_obs == 1] = [0, 255, 0]    # Snake head - green
            rgb_obs[grid_obs == 2] = [0, 128, 0]    # Snake body - dark green
            rgb_obs[grid_obs == 3] = [255, 0, 0]    # Food - red
            return self._resize_observation(rgb_obs)

    def _resize_observation(self, obs):
        """Resize observation to target size."""
        if obs.shape[:2] == self.observation_size:
            return obs.astype(np.uint8)

        try:
            from PIL import Image
            pil_img = Image.fromarray(obs)
            pil_img = pil_img.resize(
                (self.observation_size[1], self.observation_size[0]),
                Image.NEAREST
            )
            return np.array(pil_img)
        except ImportError:
            pass

        old_h, old_w = obs.shape[:2]
        new_h, new_w = self.observation_size
        h_scale = old_h / new_h
        w_scale = old_w / new_w

        if len(obs.shape) == 3:
            new_obs = np.zeros((*self.observation_size, obs.shape[2]), dtype=np.uint8)
        else:
            new_obs = np.zeros(self.observation_size, dtype=np.uint8)

        for i in range(new_h):
            for j in range(new_w):
                old_i = min(int(i * h_scale), old_h - 1)
                old_j = min(int(j * w_scale), old_w - 1)
                new_obs[i, j] = obs[old_i, old_j]

        return new_obs


def visualize_training(metrics, save_dir, env, agent):
    """Generate training visualization plots and save to disk."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('Snake DQN Training Results', fontsize=16, fontweight='bold')
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # --- 1. Episode Rewards (smoothed) ---
    ax1 = fig.add_subplot(gs[0, 0])
    rewards = metrics['episode_rewards']
    ax1.plot(rewards, alpha=0.3, color='steelblue', linewidth=0.5)
    if len(rewards) > 10:
        window = max(1, len(rewards) // 20)
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, window-1+len(smoothed)), smoothed,
                 color='darkblue', linewidth=2, label=f'Moving avg (w={window})')
        ax1.legend(fontsize=8)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Episode Rewards')
    ax1.grid(True, alpha=0.3)

    # --- 2. Episode Scores ---
    ax2 = fig.add_subplot(gs[0, 1])
    scores = metrics['episode_scores']
    ax2.plot(scores, alpha=0.3, color='green', linewidth=0.5)
    if len(scores) > 10:
        window = max(1, len(scores) // 20)
        smoothed = np.convolve(scores, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, window-1+len(smoothed)), smoothed,
                 color='darkgreen', linewidth=2, label=f'Moving avg (w={window})')
        ax2.legend(fontsize=8)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Score (food eaten)')
    ax2.set_title('Episode Scores')
    ax2.grid(True, alpha=0.3)

    # --- 3. Episode Lengths ---
    ax3 = fig.add_subplot(gs[0, 2])
    lengths = metrics['episode_lengths']
    ax3.plot(lengths, alpha=0.3, color='coral', linewidth=0.5)
    if len(lengths) > 10:
        window = max(1, len(lengths) // 20)
        smoothed = np.convolve(lengths, np.ones(window)/window, mode='valid')
        ax3.plot(range(window-1, window-1+len(smoothed)), smoothed,
                 color='red', linewidth=2, label=f'Moving avg (w={window})')
        ax3.legend(fontsize=8)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Steps')
    ax3.set_title('Episode Lengths')
    ax3.grid(True, alpha=0.3)

    # --- 4. Training Loss ---
    ax4 = fig.add_subplot(gs[1, 0])
    losses = metrics['losses']
    if losses:
        loss_steps = metrics['loss_steps']
        ax4.plot(loss_steps, losses, alpha=0.3, color='orange', linewidth=0.5)
        if len(losses) > 20:
            window = max(1, len(losses) // 20)
            smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
            ax4.plot(loss_steps[window-1:window-1+len(smoothed)], smoothed,
                     color='darkorange', linewidth=2)
        ax4.set_xlabel('Training Step')
        ax4.set_ylabel('Loss')
        ax4.set_title('Training Loss (Huber)')
        ax4.grid(True, alpha=0.3)

    # --- 5. Epsilon Decay ---
    ax5 = fig.add_subplot(gs[1, 1])
    epsilons = metrics['epsilons']
    eps_steps = metrics['epsilon_steps']
    if epsilons:
        ax5.plot(eps_steps, epsilons, color='purple', linewidth=2)
        ax5.set_xlabel('Training Step')
        ax5.set_ylabel('Epsilon')
        ax5.set_title('Exploration Rate (Epsilon)')
        ax5.grid(True, alpha=0.3)

    # --- 6. Q-values ---
    ax6 = fig.add_subplot(gs[1, 2])
    q_values = metrics['q_values']
    if q_values:
        q_steps = metrics['q_steps']
        ax6.plot(q_steps, q_values, color='teal', linewidth=1, alpha=0.5)
        if len(q_values) > 20:
            window = max(1, len(q_values) // 20)
            smoothed = np.convolve(q_values, np.ones(window)/window, mode='valid')
            ax6.plot(q_steps[window-1:window-1+len(smoothed)], smoothed,
                     color='darkcyan', linewidth=2)
        ax6.set_xlabel('Training Step')
        ax6.set_ylabel('Mean Q-value')
        ax6.set_title('Average Q-values')
        ax6.grid(True, alpha=0.3)

    # --- 7. Eval scores over time ---
    ax7 = fig.add_subplot(gs[2, 0])
    eval_steps = metrics['eval_steps']
    eval_scores = metrics['eval_avg_scores']
    eval_rewards = metrics['eval_avg_rewards']
    if eval_steps:
        ax7.plot(eval_steps, eval_scores, 'o-', color='green', linewidth=2, label='Avg Score')
        ax7_twin = ax7.twinx()
        ax7_twin.plot(eval_steps, eval_rewards, 's--', color='blue', linewidth=2, label='Avg Reward')
        ax7.set_xlabel('Training Step')
        ax7.set_ylabel('Score', color='green')
        ax7_twin.set_ylabel('Reward', color='blue')
        ax7.set_title('Evaluation Performance')
        lines1, labels1 = ax7.get_legend_handles_labels()
        lines2, labels2 = ax7_twin.get_legend_handles_labels()
        ax7.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
        ax7.grid(True, alpha=0.3)

    # --- 8. Snapshot: final game board ---
    ax8 = fig.add_subplot(gs[2, 1])
    try:
        agent.set_eval_mode()
        obs, _ = env.reset(seed=9999)
        frames = [env.renderer.render_rgb(env.engine._state, scale=1)]
        for _ in range(200):
            action = agent.act(obs, training=False)
            obs, reward, done, trunc, info = env.step(action)
            frames.append(env.renderer.render_rgb(env.engine._state, scale=1))
            if done or trunc:
                break
        ax8.imshow(frames[-1])
        ax8.set_title(f'Final Game State (score={info.get("score", 0)})')
        ax8.axis('off')
        agent.set_train_mode()
    except Exception:
        ax8.text(0.5, 0.5, 'Could not render', ha='center', va='center', transform=ax8.transAxes)
        ax8.axis('off')

    # --- 9. Score distribution histogram ---
    ax9 = fig.add_subplot(gs[2, 2])
    if scores:
        ax9.hist(scores, bins=max(1, max(scores) - min(scores) + 1),
                 color='mediumseagreen', edgecolor='darkgreen', alpha=0.8)
        ax9.axvline(np.mean(scores), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(scores):.1f}')
        ax9.legend(fontsize=8)
        ax9.set_xlabel('Score')
        ax9.set_ylabel('Count')
        ax9.set_title('Score Distribution')
        ax9.grid(True, alpha=0.3)

    plot_path = save_dir / "training_visualization.png"
    fig.savefig(str(plot_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"📊 Training visualization saved: {plot_path}")
    return plot_path


def train_simple_dqn():
    """Simple training loop with metrics tracking and visualization."""
    print("🐍 Simple Snake DQN Training")
    print("=" * 50)

    # Check dependencies
    if not TORCH_AVAILABLE:
        print("❌ This demo requires PyTorch. Install with: pip install torch")
        return

    # Set random seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Detect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    # Create environment - smaller obs for faster CPU training
    obs_size = (42, 42) if device == "cpu" else (64, 64)
    env = SimpleSnakeEnv(grid_size=(8, 8), observation_size=obs_size)
    print(f"Environment: {env.grid_size} grid, {env.obs_shape} observations")

    # Import DQN components
    try:
        from snake_rl.algos.dqn import DQNAgent

        agent = DQNAgent(
            input_channels=3,
            input_size=obs_size,
            num_actions=env.num_actions,
            lr=1e-3,
            gamma=0.99,
            buffer_size=20000,
            batch_size=32,
            target_update_freq=500,
            initial_epsilon=1.0,
            final_epsilon=0.05,
            epsilon_decay_steps=8000,
            hidden_dim=256,
            device=device
        )

        print(f"Agent: {agent.q_network.get_parameter_count():,} parameters")

    except Exception as e:
        print(f"❌ Error creating agent: {e}")
        import traceback; traceback.print_exc()
        return

    # --- Metrics tracking ---
    metrics = {
        'episode_rewards': [],
        'episode_scores': [],
        'episode_lengths': [],
        'losses': [],
        'loss_steps': np.array([], dtype=int),
        'epsilons': [],
        'epsilon_steps': np.array([], dtype=int),
        'q_values': [],
        'q_steps': np.array([], dtype=int),
        'eval_steps': [],
        'eval_avg_scores': [],
        'eval_avg_rewards': [],
    }

    # Training loop
    total_steps = 15000
    eval_frequency = 3000
    log_frequency = 1500

    step_count = 0
    episode_count = 0
    start_time = time.time()

    print(f"\n🚀 Starting training for {total_steps} steps...")

    try:
        while step_count < total_steps:
            # Start episode
            obs, _ = env.reset(seed=episode_count)
            episode_reward = 0
            episode_length = 0
            episode_count += 1

            while step_count < total_steps:
                # Select action
                action = agent.act(obs, training=True)

                # Execute action
                next_obs, reward, terminated, truncated, info = env.step(action)

                # Store experience
                agent.store_experience(
                    obs, action, reward,
                    next_obs if not (terminated or truncated) else None,
                    terminated or truncated
                )

                # Update agent
                if len(agent.replay_buffer) >= agent.batch_size:
                    update_metrics = agent.update()

                    if update_metrics:
                        metrics['losses'].append(update_metrics['loss'])
                        metrics['loss_steps'] = np.append(metrics['loss_steps'], step_count)
                        metrics['q_values'].append(update_metrics['q_value_mean'])
                        metrics['q_steps'] = np.append(metrics['q_steps'], step_count)
                        metrics['epsilons'].append(update_metrics['epsilon'])
                        metrics['epsilon_steps'] = np.append(metrics['epsilon_steps'], step_count)

                # Update counters
                obs = next_obs
                episode_reward += reward
                episode_length += 1
                step_count += 1
                agent.step_count = step_count

                # Episode end
                if terminated or truncated:
                    metrics['episode_rewards'].append(episode_reward)
                    metrics['episode_scores'].append(info.get('score', 0))
                    metrics['episode_lengths'].append(episode_length)
                    break

                # Evaluation
                if step_count % eval_frequency == 0:
                    print(f"\n📊 Evaluation at step {step_count}:")
                    eval_reward = 0
                    eval_score = 0
                    n_eval = 5

                    agent.set_eval_mode()
                    for eval_ep in range(n_eval):
                        eval_obs, _ = env.reset(seed=1000 + eval_ep)
                        ep_reward = 0

                        for eval_step in range(500):
                            eval_action = agent.act(eval_obs, training=False)
                            eval_obs, r, done, trunc, eval_info = env.step(eval_action)
                            ep_reward += r

                            if done or trunc:
                                eval_score += eval_info.get('score', 0)
                                break

                        eval_reward += ep_reward
                    agent.set_train_mode()

                    avg_reward = eval_reward / n_eval
                    avg_score = eval_score / n_eval
                    metrics['eval_steps'].append(step_count)
                    metrics['eval_avg_scores'].append(avg_score)
                    metrics['eval_avg_rewards'].append(avg_reward)
                    print(f"   Avg reward: {avg_reward:.2f}, Avg score: {avg_score:.1f}")

            # Periodic logging
            if step_count % log_frequency < episode_length or step_count >= total_steps:
                elapsed = time.time() - start_time
                steps_per_sec = step_count / max(elapsed, 1e-6)
                epsilon = agent.epsilon_scheduler.get_epsilon()
                recent_scores = metrics['episode_scores'][-20:]
                avg_recent = np.mean(recent_scores) if recent_scores else 0

                print(f"Step {step_count:6d}/{total_steps} | Ep {episode_count:4d} | "
                      f"Score: {info.get('score', 0):2d} | Avg(20): {avg_recent:.1f} | "
                      f"Eps: {epsilon:.3f} | {steps_per_sec:.0f} stp/s")

    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")

    # Final summary
    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"📊 Training Summary:")
    print(f"   Total steps:      {step_count:,}")
    print(f"   Total episodes:   {episode_count:,}")
    print(f"   Training time:    {elapsed:.1f}s")
    print(f"   Steps per second: {step_count/max(elapsed, 1e-6):.0f}")
    if metrics['episode_scores']:
        print(f"   Best score:       {max(metrics['episode_scores'])}")
        print(f"   Avg score (last 20): {np.mean(metrics['episode_scores'][-20:]):.1f}")

    # Save model
    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)

    model_path = save_dir / "simple_demo_model.pt"
    agent.save(str(model_path))
    print(f"💾 Model saved: {model_path}")

    # Final evaluation
    print(f"\n🎯 Final evaluation (10 episodes):")
    total_reward = 0
    total_score = 0
    n_final = 10
    agent.set_eval_mode()

    for ep in range(n_final):
        obs, _ = env.reset(seed=2000 + ep)
        ep_reward = 0

        for step in range(500):
            action = agent.act(obs, training=False)
            obs, reward, done, trunc, info = env.step(action)
            ep_reward += reward

            if done or trunc:
                total_score += info.get('score', 0)
                break

        total_reward += ep_reward

    print(f"   Average reward: {total_reward/n_final:.2f}")
    print(f"   Average score:  {total_score/n_final:.1f}")

    # Generate visualization
    print(f"\n🎨 Generating training visualization...")
    try:
        visualize_training(metrics, save_dir, env, agent)
    except Exception as e:
        print(f"⚠️ Visualization error: {e}")
        import traceback; traceback.print_exc()

    print(f"\n✅ Training complete!")


if __name__ == "__main__":
    train_simple_dqn()
