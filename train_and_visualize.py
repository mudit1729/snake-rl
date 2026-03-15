#!/usr/bin/env python3
"""
Train DQN and GRPO agents on Snake, generate comparison videos.

This script:
1. Trains a DQN agent with checkpoints saved at key intervals.
2. Trains a GRPO agent with checkpoints saved at key intervals.
3. Generates per-algorithm progression videos showing learning over time.
4. Generates a side-by-side comparison video of DQN vs GRPO.
5. Generates a multi-panel comparison plot of training metrics.
"""

import os
import sys
import time
import random
import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Force headless pygame / SDL (must be set BEFORE any pygame import)
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import numpy as np

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath('.'))

import torch
from torch.utils.tensorboard import SummaryWriter
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from snake_rl.sim.engine import SnakeEngine, Action
from snake_rl.sim.renderer import SnakeRenderer
from snake_rl.algos.dqn import DQNAgent
from snake_rl.algos.grpo import GRPOAgent
from snake_rl.algos.mcts import MCTSPlanner, make_mcts_config


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SimpleSnakeEnv:
    """Simplified Snake environment without Gymnasium dependency."""

    def __init__(self, grid_size=(10, 10), observation_size=(84, 84)):
        self.grid_size = grid_size
        self.observation_size = observation_size
        self.engine = SnakeEngine(grid_size=grid_size)
        self.renderer = SnakeRenderer()
        self.num_actions = 4
        self.obs_shape = (*observation_size, 3)

    def reset(self, seed=None):
        state = self.engine.reset(seed=seed)
        obs = self._get_observation(state)
        return obs, {"score": state.score, "steps": state.steps}

    def step(self, action):
        state, reward, done, info = self.engine.step(action)
        obs = self._get_observation(state)
        return obs, reward, done, False, info

    def _get_observation(self, state):
        try:
            rgb_obs = self.renderer.render_rgb(state, scale=1)
            return self._resize_observation(rgb_obs)
        except Exception:
            grid_obs = self.renderer.render_array(state)
            h, w = grid_obs.shape
            rgb_obs = np.zeros((h, w, 3), dtype=np.uint8)
            rgb_obs[grid_obs == 1] = [0, 255, 0]
            rgb_obs[grid_obs == 2] = [0, 128, 0]
            rgb_obs[grid_obs == 3] = [255, 0, 0]
            return self._resize_observation(rgb_obs)

    def _resize_observation(self, obs):
        if obs.shape[:2] == self.observation_size:
            return obs.astype(np.uint8)
        try:
            from PIL import Image as _Img
            pil_img = _Img.fromarray(obs)
            pil_img = pil_img.resize(
                (self.observation_size[1], self.observation_size[0]),
                _Img.NEAREST,
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
                new_obs[i, j] = obs[min(int(i * h_scale), old_h - 1),
                                    min(int(j * w_scale), old_w - 1)]
        return new_obs


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _make_planner(mcts_config: Optional[Dict[str, Any]]) -> Optional[MCTSPlanner]:
    config = make_mcts_config(mcts_config)
    if config is None or not config.enabled:
        return None
    return MCTSPlanner(config)


def _select_action(agent, env: SimpleSnakeEnv, obs: np.ndarray, training: bool, planner: Optional[MCTSPlanner]) -> int:
    if planner is None:
        return agent.act(obs, training=training)
    return planner.select_action(env=env, agent=agent, observation=obs, training=training)


def _log_score_averages(
    tb_writer: SummaryWriter,
    prefix: str,
    scores: List[float],
    x_value: int,
    windows: Tuple[int, ...] = (10, 25, 50, 100),
) -> None:
    if not scores:
        return

    tb_writer.add_scalar(f"{prefix}/avg_score_running", float(np.mean(scores)), x_value)
    for window in windows:
        if len(scores) >= window:
            tb_writer.add_scalar(
                f"{prefix}/avg_score_{window}",
                float(np.mean(scores[-window:])),
                x_value,
            )


# ---------------------------------------------------------------------------
# DQN Training
# ---------------------------------------------------------------------------

def train_dqn(
    device: str,
    save_dir: Path,
    total_steps: int = 50_000,
    checkpoint_steps: Optional[List[int]] = None,
    log_frequency: int = 2000,
    tb_log_dir: Optional[Path] = None,
    hidden_dim: int = 128,
    encoder_type: str = "small_spatial",
    mcts_config: Optional[Dict[str, Any]] = None,
    resume_checkpoint: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train a DQN agent and save checkpoints. Returns training metrics."""
    print("\n" + "=" * 60)
    print("  DQN Training")
    print("=" * 60)

    save_dir.mkdir(parents=True, exist_ok=True)

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    obs_size = (42, 42)
    grid_size = (10, 10)
    env = SimpleSnakeEnv(grid_size=grid_size, observation_size=obs_size)
    planner = _make_planner(mcts_config)

    agent = DQNAgent(
        input_channels=3,
        input_size=obs_size,
        num_actions=env.num_actions,
        lr=1e-3,
        gamma=0.99,
        buffer_size=50000,
        batch_size=64,
        target_update_freq=1000,
        initial_epsilon=1.0,
        final_epsilon=0.05,
        epsilon_decay_steps=25000,
        hidden_dim=hidden_dim,
        encoder_type=encoder_type,
        device=device,
    )
    print(f"  Device: {device}")
    print(f"  Parameters: {agent.q_network.get_parameter_count():,}")
    if planner is not None:
        print(f"  MCTS: {asdict(planner.config)}")
    if resume_checkpoint is not None:
        agent.load(str(resume_checkpoint))
        print(f"  Resumed from: {resume_checkpoint}")

    if checkpoint_steps is None:
        checkpoint_steps = [0, 5000, 10000, 20000, 35000, 50000]

    metrics: Dict[str, Any] = {
        "episode_scores": [],
        "episode_rewards": [],
        "losses": [],
        "loss_steps": [],
        "eval_scores_at_checkpoints": {},
    }

    if resume_checkpoint is None:
        # Save initial (random) checkpoint
        agent.save(str(save_dir / "checkpoint_0.pt"))
        print("  Saved checkpoint at step 0 (random)")

    # TensorBoard
    tb_dir = tb_log_dir or (Path("logs") / "tensorboard" / "dqn")
    tb_writer = SummaryWriter(str(tb_dir))
    print(f"  TensorBoard logging to: {tb_dir}")

    step_count = agent.step_count
    initial_step_count = step_count
    episode_count = 0
    start_time = time.time()

    if step_count >= total_steps:
        print(f"  Checkpoint already at step {step_count}, target is {total_steps}.")
        tb_writer.close()
        return metrics

    while step_count < total_steps:
        obs, _ = env.reset(seed=episode_count % 1000)
        episode_reward = 0.0
        episode_count += 1

        while step_count < total_steps:
            action = _select_action(agent, env, obs, training=True, planner=planner)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store_experience(
                obs, action, reward,
                next_obs if not done else None,
                done,
            )

            if len(agent.replay_buffer) >= agent.batch_size:
                update_metrics = agent.update()
                if update_metrics:
                    metrics["losses"].append(update_metrics["loss"])
                    metrics["loss_steps"].append(step_count)
                    if step_count % 100 == 0:
                        tb_writer.add_scalar("dqn/loss", update_metrics["loss"], step_count)
                        tb_writer.add_scalar("dqn/q_value_mean", update_metrics.get("q_value_mean", 0), step_count)
                        tb_writer.add_scalar("dqn/epsilon", update_metrics.get("epsilon", 0), step_count)

            obs = next_obs
            episode_reward += reward
            step_count += 1
            agent.step_count = step_count

            # Checkpoint saving
            if step_count in checkpoint_steps:
                ckpt_path = save_dir / f"checkpoint_{step_count}.pt"
                agent.save(str(ckpt_path))
                print(f"  Saved checkpoint at step {step_count}")

            if done:
                score = info.get("score", 0)
                metrics["episode_scores"].append(score)
                metrics["episode_rewards"].append(episode_reward)
                tb_writer.add_scalar("dqn/episode_score", score, step_count)
                tb_writer.add_scalar("dqn/episode_reward", episode_reward, step_count)
                _log_score_averages(
                    tb_writer,
                    "dqn",
                    metrics["episode_scores"],
                    step_count,
                    windows=(10, 25, 50),
                )
                break

        # Periodic logging
        if step_count % log_frequency < 200 or step_count >= total_steps:
            elapsed = time.time() - start_time
            sps = (step_count - initial_step_count) / max(elapsed, 1e-6)
            recent = metrics["episode_scores"][-50:]
            avg = np.mean(recent) if recent else 0
            eps = agent.epsilon_scheduler.get_epsilon()
            print(f"  Step {step_count:>7d}/{total_steps} | Ep {episode_count:>5d} | "
                  f"Avg(50): {avg:.1f} | Eps: {eps:.3f} | {sps:.0f} stp/s")

    tb_writer.close()

    elapsed = time.time() - start_time
    print(f"  DQN training complete in {elapsed:.1f}s "
          f"({(step_count - initial_step_count) / max(elapsed, 1e-6):.0f} stp/s)")

    # Save metrics
    serialisable = {
        "episode_scores": metrics["episode_scores"],
        "episode_rewards": metrics["episode_rewards"],
        "losses": metrics["losses"],
        "loss_steps": metrics["loss_steps"],
    }
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(serialisable, f)

    return metrics


# ---------------------------------------------------------------------------
# GRPO Training
# ---------------------------------------------------------------------------

def train_grpo(
    device: str,
    save_dir: Path,
    total_episodes: int = 1000,
    checkpoint_episodes: Optional[List[int]] = None,
    log_frequency: int = 50,
    trajectories_per_update: int = 8,
    tb_log_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train a GRPO agent and save checkpoints. Returns training metrics."""
    print("\n" + "=" * 60)
    print("  GRPO Training")
    print("=" * 60)

    save_dir.mkdir(parents=True, exist_ok=True)

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    obs_size = (42, 42)
    grid_size = (10, 10)
    env = SimpleSnakeEnv(grid_size=grid_size, observation_size=obs_size)

    agent = GRPOAgent(
        input_channels=3,
        input_size=obs_size,
        num_actions=env.num_actions,
        lr=3e-4,
        gamma=0.99,
        group_size=4,
        clip_epsilon=0.2,
        entropy_coef=0.02,
        value_coef=0.5,
        base_channels=16,
        hidden_dim=64,
        device=device,
    )
    print(f"  Device: {device}")
    print(f"  Parameters: {agent.network.get_parameter_count():,}")

    if checkpoint_episodes is None:
        checkpoint_episodes = [0, 50, 100, 250, 500, 750, 1000]

    metrics: Dict[str, Any] = {
        "episode_scores": [],
        "episode_rewards": [],
        "policy_losses": [],
        "policy_loss_episodes": [],
    }

    # Save initial (random) checkpoint
    agent.save(str(save_dir / "checkpoint_ep0.pt"))
    print("  Saved checkpoint at episode 0 (random)")

    # TensorBoard
    tb_dir = tb_log_dir or (Path("logs") / "tensorboard" / "grpo")
    tb_writer = SummaryWriter(str(tb_dir))
    print(f"  TensorBoard logging to: {tb_dir}")

    start_time = time.time()
    pending_trajectories: List[List[Any]] = []

    for episode in range(1, total_episodes + 1):
        trajectory = agent.collect_trajectory(env, max_steps=500)
        pending_trajectories.append(trajectory)

        # Extract episode stats
        ep_reward = sum(step.reward for step in trajectory)
        # The score is approximately the number of positive rewards (food eaten)
        ep_score = sum(1 for step in trajectory if step.reward > 0)
        metrics["episode_scores"].append(ep_score)
        metrics["episode_rewards"].append(ep_reward)

        tb_writer.add_scalar("grpo/episode_score", ep_score, episode)
        tb_writer.add_scalar("grpo/episode_reward", ep_reward, episode)
        _log_score_averages(
            tb_writer,
            "grpo",
            metrics["episode_scores"],
            episode,
            windows=(10, 25, 50, 100),
        )

        # Update on a small batch of fresh on-policy trajectories.
        if (
            len(pending_trajectories) >= trajectories_per_update
            or episode == total_episodes
        ):
            batch_steps = sum(len(traj) for traj in pending_trajectories)
            update_info = agent.update(
                pending_trajectories,
                epochs=4,
                mini_batch_size=min(64, max(1, batch_steps)),
            )
            pending_trajectories.clear()

            if update_info and "policy_loss" in update_info:
                metrics["policy_losses"].append(update_info["policy_loss"])
                metrics["policy_loss_episodes"].append(episode)
                tb_writer.add_scalar("grpo/policy_loss", update_info["policy_loss"], episode)
                if "entropy" in update_info:
                    tb_writer.add_scalar("grpo/entropy", update_info["entropy"], episode)

        # Checkpoint saving
        if episode in checkpoint_episodes:
            ckpt_path = save_dir / f"checkpoint_ep{episode}.pt"
            agent.save(str(ckpt_path))
            print(f"  Saved checkpoint at episode {episode}")

        # Periodic logging
        if episode % log_frequency == 0:
            elapsed = time.time() - start_time
            recent = metrics["episode_scores"][-100:]
            avg = np.mean(recent) if recent else 0
            print(f"  Episode {episode:>5d}/{total_episodes} | "
                  f"Avg(100): {avg:.2f} | Reward: {ep_reward:.2f} | "
                  f"{elapsed:.0f}s elapsed")

    tb_writer.close()
    elapsed = time.time() - start_time
    print(f"  GRPO training complete in {elapsed:.1f}s")

    serialisable = {
        "episode_scores": metrics["episode_scores"],
        "episode_rewards": metrics["episode_rewards"],
        "policy_losses": metrics["policy_losses"],
        "policy_loss_episodes": metrics["policy_loss_episodes"],
    }
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(serialisable, f)

    return metrics


# ---------------------------------------------------------------------------
# Video generation helpers
# ---------------------------------------------------------------------------

FRAME_SIZE = 480  # pixels, square


def _get_font(size: int = 16):
    """Try to load a TrueType font; fall back to the default bitmap font."""
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        pass
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except Exception:
        pass
    try:
        return ImageFont.truetype("/System/Library/Fonts/SFNSMono.ttf", size)
    except Exception:
        pass
    return ImageFont.load_default()


def create_title_frame(
    text: str,
    width: int = FRAME_SIZE,
    height: int = FRAME_SIZE,
    bg_color: Tuple[int, int, int] = (20, 20, 40),
    text_color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Create a solid-colour frame with centred text."""
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    font = _get_font(28)

    # Split text into lines and draw each centred
    lines = text.split("\n")
    line_height = 36
    total_height = line_height * len(lines)
    y_start = (height - total_height) // 2

    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        x = (width - tw) // 2
        y = y_start + i * line_height
        draw.text((x, y), line, fill=text_color, font=font)

    return np.array(img)


def add_overlay(
    frame: np.ndarray,
    text_lines: List[str],
    position: str = "top",
) -> np.ndarray:
    """Add text overlay lines to a frame. Returns a new array."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    font = _get_font(14)

    y = 6 if position == "top" else frame.shape[0] - 20 * len(text_lines) - 6

    for line in text_lines:
        # Draw shadow then text for readability
        draw.text((7, y + 1), line, fill=(0, 0, 0), font=font)
        draw.text((6, y), line, fill=(255, 255, 255), font=font)
        y += 18

    return np.array(img)


def _render_game_frame(env: SimpleSnakeEnv) -> np.ndarray:
    """Render the current game state as a FRAME_SIZE x FRAME_SIZE RGB array."""
    try:
        raw = env.renderer.render_rgb(env.engine._state, scale=1)
    except Exception:
        arr = env.renderer.render_array(env.engine._state)
        h, w = arr.shape
        raw = np.zeros((h, w, 3), dtype=np.uint8)
        raw[arr == 1] = [0, 255, 0]
        raw[arr == 2] = [0, 128, 0]
        raw[arr == 3] = [255, 0, 0]

    pil = Image.fromarray(raw)
    pil = pil.resize((FRAME_SIZE, FRAME_SIZE), Image.NEAREST)
    return np.array(pil)


def record_episode(
    env: SimpleSnakeEnv,
    agent,
    seed: int = 42,
    max_steps: int = 500,
    planner: Optional[MCTSPlanner] = None,
) -> Tuple[List[np.ndarray], int, float]:
    """
    Play one episode with the agent in eval mode and record frames.

    Returns (frames, score, total_reward).
    """
    obs, _ = env.reset(seed=seed)
    frames: List[np.ndarray] = [_render_game_frame(env)]
    total_reward = 0.0
    score = 0

    for step in range(max_steps):
        action = _select_action(agent, env, obs, training=False, planner=planner)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        score = info.get("score", score)
        frames.append(_render_game_frame(env))
        if terminated or truncated:
            break

    return frames, score, total_reward


def _get_video_writer(output_path: str, fps: int = 10):
    """Return an imageio writer, or fall back to matplotlib animation saver."""
    try:
        import imageio.v2 as imageio
        return imageio.get_writer(
            output_path,
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=1,
        )
    except Exception:
        pass
    try:
        import imageio
        return imageio.get_writer(
            output_path,
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=1,
        )
    except Exception:
        pass
    return None


class _FallbackWriter:
    """Collect frames and save with matplotlib animation pillow writer."""

    def __init__(self, output_path: str, fps: int = 10):
        self.output_path = output_path
        self.fps = fps
        self.frames: List[np.ndarray] = []

    def append_data(self, frame: np.ndarray):
        self.frames.append(frame)

    def close(self):
        if not self.frames:
            return
        from matplotlib.animation import FuncAnimation, PillowWriter
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.axis("off")
        im = ax.imshow(self.frames[0])

        def update(i):
            im.set_data(self.frames[i])
            return [im]

        anim = FuncAnimation(fig, update, frames=len(self.frames), interval=1000 // self.fps, blit=True)
        # Try mp4 with ffmpeg, fall back to gif with pillow
        try:
            anim.save(self.output_path, writer="ffmpeg", fps=self.fps)
        except Exception:
            gif_path = self.output_path.replace(".mp4", ".gif")
            anim.save(gif_path, writer=PillowWriter(fps=self.fps))
            print(f"    (saved as GIF instead: {gif_path})")
        plt.close(fig)


def _open_writer(output_path: str, fps: int = 10):
    writer = _get_video_writer(output_path, fps)
    if writer is not None:
        return writer
    return _FallbackWriter(output_path, fps)


# ---------------------------------------------------------------------------
# Progression video for a single algorithm
# ---------------------------------------------------------------------------

def generate_video(
    checkpoint_dir: Path,
    agent_class,
    agent_kwargs: Dict[str, Any],
    env_kwargs: Dict[str, Any],
    output_path: str,
    algorithm_name: str,
    checkpoint_names: List[str],
    checkpoint_labels: List[str],
    mcts_config: Optional[Dict[str, Any]] = None,
):
    """
    Generate a progression video for one algorithm.

    checkpoint_names: list of filenames inside checkpoint_dir
    checkpoint_labels: human-readable label per checkpoint (e.g. "Step 5000")
    """
    print(f"\n  Generating {algorithm_name} progression video ...")
    env = SimpleSnakeEnv(**env_kwargs)
    planner = _make_planner(mcts_config)
    writer = _open_writer(output_path)
    fps = 10
    title_card_frames = fps  # 1 second of title card

    for ckpt_file, label in zip(checkpoint_names, checkpoint_labels):
        ckpt_path = checkpoint_dir / ckpt_file
        if not ckpt_path.exists():
            print(f"    Checkpoint not found, skipping: {ckpt_path}")
            continue

        # Title card
        title = f"{algorithm_name}\n{label}"
        title_frame = create_title_frame(title)
        for _ in range(title_card_frames):
            writer.append_data(title_frame)

        # Load agent
        agent = agent_class(**agent_kwargs)
        agent.load(str(ckpt_path))
        agent.set_eval_mode()

        # Run 3 episodes, pick best score
        best_frames = None
        best_score = -1
        for s in [42, 43, 44]:
            frames, score, _ = record_episode(env, agent, seed=s, planner=planner)
            if score > best_score:
                best_score = score
                best_frames = frames

        # Write frames with overlay
        for idx, frame in enumerate(best_frames):
            frame = add_overlay(frame, [
                f"{algorithm_name} | {label}",
                f"Score: {best_score}  Frame: {idx}",
            ])
            writer.append_data(frame)

    writer.close()
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Side-by-side comparison video
# ---------------------------------------------------------------------------

def generate_comparison_video(
    dqn_dir: Path,
    grpo_dir: Path,
    dqn_agent_kwargs: Dict[str, Any],
    grpo_agent_kwargs: Dict[str, Any],
    env_kwargs: Dict[str, Any],
    output_path: str,
    stage_labels: List[str],
    dqn_ckpts: List[str],
    grpo_ckpts: List[str],
    dqn_mcts_config: Optional[Dict[str, Any]] = None,
    grpo_mcts_config: Optional[Dict[str, Any]] = None,
):
    """Generate side-by-side DQN vs GRPO comparison video."""
    print("\n  Generating comparison video ...")
    env = SimpleSnakeEnv(**env_kwargs)
    dqn_planner = _make_planner(dqn_mcts_config)
    grpo_planner = _make_planner(grpo_mcts_config)
    writer = _open_writer(output_path)
    fps = 10
    title_card_frames = fps
    combined_width = FRAME_SIZE * 2 + 20  # 20 px gap

    for stage_label, dqn_file, grpo_file in zip(stage_labels, dqn_ckpts, grpo_ckpts):
        dqn_path = dqn_dir / dqn_file
        grpo_path = grpo_dir / grpo_file

        if not dqn_path.exists() or not grpo_path.exists():
            print(f"    Skipping stage '{stage_label}' (missing checkpoint)")
            continue

        # Title card
        title_frame = create_title_frame(
            f"DQN vs GRPO\n{stage_label}",
            width=combined_width,
        )
        for _ in range(title_card_frames):
            writer.append_data(title_frame)

        # Load agents
        dqn_agent = DQNAgent(**dqn_agent_kwargs)
        dqn_agent.load(str(dqn_path))
        dqn_agent.set_eval_mode()

        grpo_agent = GRPOAgent(**grpo_agent_kwargs)
        grpo_agent.load(str(grpo_path))
        grpo_agent.set_eval_mode()

        # Best of 3 for each
        def _best_episode(agent, planner):
            best_f, best_s = None, -1
            for s in [42, 43, 44]:
                f, sc, _ = record_episode(env, agent, seed=s, planner=planner)
                if sc > best_s:
                    best_s = sc
                    best_f = f
            return best_f, best_s

        dqn_frames, dqn_score = _best_episode(dqn_agent, dqn_planner)
        grpo_frames, grpo_score = _best_episode(grpo_agent, grpo_planner)

        # Equalise length
        max_len = max(len(dqn_frames), len(grpo_frames))
        while len(dqn_frames) < max_len:
            dqn_frames.append(dqn_frames[-1])
        while len(grpo_frames) < max_len:
            grpo_frames.append(grpo_frames[-1])

        for idx in range(max_len):
            left = add_overlay(dqn_frames[idx], [
                f"DQN | {stage_label}",
                f"Score: {dqn_score}",
            ])
            right = add_overlay(grpo_frames[idx], [
                f"GRPO | {stage_label}",
                f"Score: {grpo_score}",
            ])
            gap = np.zeros((FRAME_SIZE, 20, 3), dtype=np.uint8)
            combined = np.concatenate([left, gap, right], axis=1)
            writer.append_data(combined)

    writer.close()
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Comparison plot
# ---------------------------------------------------------------------------

def generate_comparison_plot(
    dqn_metrics: Dict[str, Any],
    grpo_metrics: Dict[str, Any],
    dqn_dir: Path,
    grpo_dir: Path,
    save_path: str,
):
    """Generate a 6-panel comparison figure."""
    print("\n  Generating comparison plot ...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("DQN vs GRPO Training Comparison", fontsize=16, fontweight="bold")

    # --- Panel 1: DQN scores ---
    ax = axes[0, 0]
    scores = dqn_metrics["episode_scores"]
    ax.plot(scores, alpha=0.25, color="steelblue", linewidth=0.5)
    if len(scores) > 20:
        w = max(1, len(scores) // 30)
        sm = np.convolve(scores, np.ones(w) / w, mode="valid")
        ax.plot(range(w - 1, w - 1 + len(sm)), sm, color="darkblue", linewidth=2)
    ax.set_title("DQN Scores over Training")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.3)

    # --- Panel 2: GRPO scores ---
    ax = axes[0, 1]
    scores_g = grpo_metrics["episode_scores"]
    ax.plot(scores_g, alpha=0.25, color="coral", linewidth=0.5)
    if len(scores_g) > 20:
        w = max(1, len(scores_g) // 30)
        sm = np.convolve(scores_g, np.ones(w) / w, mode="valid")
        ax.plot(range(w - 1, w - 1 + len(sm)), sm, color="darkred", linewidth=2)
    ax.set_title("GRPO Scores over Training")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Eval scores at checkpoints ---
    ax = axes[0, 2]
    # Run quick eval at each checkpoint
    env_kwargs = {"grid_size": (10, 10), "observation_size": (42, 42)}
    env = SimpleSnakeEnv(**env_kwargs)

    dqn_ckpt_steps = [0, 5000, 10000, 20000, 35000, 50000]
    dqn_eval_scores = []
    dqn_agent_kwargs = {
        "input_channels": 3, "input_size": (42, 42), "num_actions": 4,
        "lr": 1e-3, "gamma": 0.99, "buffer_size": 50000, "batch_size": 64,
        "target_update_freq": 1000, "initial_epsilon": 1.0, "final_epsilon": 0.05,
        "epsilon_decay_steps": 25000, "hidden_dim": 128, "encoder_type": "small_spatial",
        "device": "cpu",
    }
    for step_val in dqn_ckpt_steps:
        p = dqn_dir / f"checkpoint_{step_val}.pt"
        if p.exists():
            a = DQNAgent(**dqn_agent_kwargs)
            a.load(str(p))
            a.set_eval_mode()
            sc_list = []
            for s in [42, 43, 44]:
                _, sc, _ = record_episode(env, a, seed=s, max_steps=300)
                sc_list.append(sc)
            dqn_eval_scores.append(np.mean(sc_list))
        else:
            dqn_eval_scores.append(0)

    grpo_ckpt_eps = [0, 50, 100, 250, 500, 750, 1000]
    grpo_eval_scores = []
    grpo_agent_kwargs = {
        "input_channels": 3, "input_size": (42, 42), "num_actions": 4,
        "lr": 3e-4, "gamma": 0.99, "group_size": 4, "clip_epsilon": 0.2,
        "entropy_coef": 0.02, "value_coef": 0.5,
        "base_channels": 16, "hidden_dim": 64, "device": "cpu",
    }
    for ep_val in grpo_ckpt_eps:
        p = grpo_dir / f"checkpoint_ep{ep_val}.pt"
        if p.exists():
            a = GRPOAgent(**grpo_agent_kwargs)
            a.load(str(p))
            a.set_eval_mode()
            sc_list = []
            for s in [42, 43, 44]:
                _, sc, _ = record_episode(env, a, seed=s, max_steps=300)
                sc_list.append(sc)
            grpo_eval_scores.append(np.mean(sc_list))
        else:
            grpo_eval_scores.append(0)

    ax.plot(range(len(dqn_ckpt_steps)), dqn_eval_scores, "o-", color="steelblue",
            linewidth=2, label="DQN")
    ax.set_xticks(range(len(dqn_ckpt_steps)))
    ax.set_xticklabels([str(s) for s in dqn_ckpt_steps], rotation=45, fontsize=7)
    ax2 = ax.twiny()
    ax2.plot(range(len(grpo_ckpt_eps)), grpo_eval_scores, "s--", color="coral",
             linewidth=2, label="GRPO")
    ax2.set_xticks(range(len(grpo_ckpt_eps)))
    ax2.set_xticklabels([str(e) for e in grpo_ckpt_eps], rotation=45, fontsize=7)
    ax2.set_xlabel("GRPO Episode", fontsize=8)
    ax.set_xlabel("DQN Step", fontsize=8)
    ax.set_ylabel("Eval Score")
    ax.set_title("Checkpoint Eval Scores")
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 4: DQN loss curve ---
    ax = axes[1, 0]
    losses = dqn_metrics.get("losses", [])
    loss_steps = dqn_metrics.get("loss_steps", [])
    if losses:
        ax.plot(loss_steps, losses, alpha=0.2, color="orange", linewidth=0.5)
        if len(losses) > 30:
            w = max(1, len(losses) // 30)
            sm = np.convolve(losses, np.ones(w) / w, mode="valid")
            ax.plot(loss_steps[w - 1:w - 1 + len(sm)], sm, color="darkorange", linewidth=2)
    ax.set_title("DQN Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    # --- Panel 5: GRPO policy loss curve ---
    ax = axes[1, 1]
    pl = grpo_metrics.get("policy_losses", [])
    pl_eps = grpo_metrics.get("policy_loss_episodes", [])
    if pl:
        ax.plot(pl_eps, pl, alpha=0.2, color="mediumpurple", linewidth=0.5)
        if len(pl) > 30:
            w = max(1, len(pl) // 30)
            sm = np.convolve(pl, np.ones(w) / w, mode="valid")
            ax.plot(pl_eps[w - 1:w - 1 + len(sm)], sm, color="purple", linewidth=2)
    ax.set_title("GRPO Policy Loss")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Policy Loss")
    ax.grid(True, alpha=0.3)

    # --- Panel 6: Final score distributions ---
    ax = axes[1, 2]
    dqn_last = dqn_metrics["episode_scores"][-200:] if len(dqn_metrics["episode_scores"]) >= 200 else dqn_metrics["episode_scores"]
    grpo_last = grpo_metrics["episode_scores"][-200:] if len(grpo_metrics["episode_scores"]) >= 200 else grpo_metrics["episode_scores"]
    bins = max(1, max(max(dqn_last, default=0), max(grpo_last, default=0)) + 1)
    bins = min(bins, 30)
    ax.hist(dqn_last, bins=bins, alpha=0.6, color="steelblue", label=f"DQN (mean={np.mean(dqn_last):.1f})")
    ax.hist(grpo_last, bins=bins, alpha=0.6, color="coral", label=f"GRPO (mean={np.mean(grpo_last):.1f})")
    ax.legend(fontsize=8)
    ax.set_title("Final Score Distributions (last 200 ep)")
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Snake RL: DQN + GRPO Training & Visualization")
    print("=" * 60)

    device = _detect_device()
    print(f"  Device: {device}")

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    dqn_dir = output_dir / "dqn_checkpoints"
    grpo_dir = output_dir / "grpo_checkpoints"

    # ---- Train ----
    dqn_metrics = train_dqn(device, dqn_dir)
    grpo_metrics = train_grpo(device, grpo_dir)

    # ---- Common kwargs for agent construction during video generation ----
    obs_size = (42, 42)
    env_kwargs = {"grid_size": (10, 10), "observation_size": obs_size}

    dqn_agent_kwargs = {
        "input_channels": 3,
        "input_size": obs_size,
        "num_actions": 4,
        "lr": 1e-3,
        "gamma": 0.99,
        "buffer_size": 50000,
        "batch_size": 64,
        "target_update_freq": 1000,
        "initial_epsilon": 1.0,
        "final_epsilon": 0.05,
        "epsilon_decay_steps": 25000,
        "hidden_dim": 128,
        "encoder_type": "small_spatial",
        "device": "cpu",
    }

    grpo_agent_kwargs = {
        "input_channels": 3,
        "input_size": obs_size,
        "num_actions": 4,
        "lr": 3e-4,
        "gamma": 0.99,
        "group_size": 4,
        "clip_epsilon": 0.2,
        "entropy_coef": 0.02,
        "value_coef": 0.5,
        "base_channels": 16,
        "hidden_dim": 64,
        "device": "cpu",
    }

    # ---- DQN progression video ----
    dqn_ckpt_names = [
        "checkpoint_0.pt", "checkpoint_5000.pt", "checkpoint_10000.pt",
        "checkpoint_20000.pt", "checkpoint_35000.pt", "checkpoint_50000.pt",
    ]
    dqn_ckpt_labels = [
        "Step 0 (random)", "Step 5,000", "Step 10,000",
        "Step 20,000", "Step 35,000", "Step 50,000",
    ]

    generate_video(
        checkpoint_dir=dqn_dir,
        agent_class=DQNAgent,
        agent_kwargs=dqn_agent_kwargs,
        env_kwargs=env_kwargs,
        output_path=str(output_dir / "dqn_progression.mp4"),
        algorithm_name="DQN",
        checkpoint_names=dqn_ckpt_names,
        checkpoint_labels=dqn_ckpt_labels,
    )

    # ---- GRPO progression video ----
    grpo_ckpt_names = [
        "checkpoint_ep0.pt", "checkpoint_ep50.pt", "checkpoint_ep100.pt",
        "checkpoint_ep250.pt", "checkpoint_ep500.pt", "checkpoint_ep750.pt",
        "checkpoint_ep1000.pt",
    ]
    grpo_ckpt_labels = [
        "Episode 0 (random)", "Episode 50", "Episode 100",
        "Episode 250", "Episode 500", "Episode 750", "Episode 1,000",
    ]

    generate_video(
        checkpoint_dir=grpo_dir,
        agent_class=GRPOAgent,
        agent_kwargs=grpo_agent_kwargs,
        env_kwargs=env_kwargs,
        output_path=str(output_dir / "grpo_progression.mp4"),
        algorithm_name="GRPO",
        checkpoint_names=grpo_ckpt_names,
        checkpoint_labels=grpo_ckpt_labels,
    )

    # ---- Comparison video ----
    # Match stages: random / early / mid / late / final
    comparison_stages = [
        "Random",
        "Early",
        "Mid",
        "Late",
        "Final",
    ]
    comparison_dqn = [
        "checkpoint_0.pt",
        "checkpoint_5000.pt",
        "checkpoint_10000.pt",
        "checkpoint_35000.pt",
        "checkpoint_50000.pt",
    ]
    comparison_grpo = [
        "checkpoint_ep0.pt",
        "checkpoint_ep50.pt",
        "checkpoint_ep250.pt",
        "checkpoint_ep750.pt",
        "checkpoint_ep1000.pt",
    ]

    generate_comparison_video(
        dqn_dir=dqn_dir,
        grpo_dir=grpo_dir,
        dqn_agent_kwargs=dqn_agent_kwargs,
        grpo_agent_kwargs=grpo_agent_kwargs,
        env_kwargs=env_kwargs,
        output_path=str(output_dir / "comparison.mp4"),
        stage_labels=comparison_stages,
        dqn_ckpts=comparison_dqn,
        grpo_ckpts=comparison_grpo,
    )

    # ---- Comparison plot ----
    generate_comparison_plot(
        dqn_metrics=dqn_metrics,
        grpo_metrics=grpo_metrics,
        dqn_dir=dqn_dir,
        grpo_dir=grpo_dir,
        save_path=str(output_dir / "training_comparison.png"),
    )

    print("\n" + "=" * 60)
    print("  All done!")
    print(f"  Outputs in: {output_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
