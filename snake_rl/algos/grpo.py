"""
Group Relative Policy Optimization (GRPO) for Snake Game

GRPO is a policy gradient method that computes advantages relative to a group
of sampled actions, eliminating the need for a separate value baseline network.

Key idea: For each state, sample a GROUP of actions (e.g., 8-16), evaluate each
via 1-step lookahead with value bootstrap, then compute advantages relative to
the group mean. The policy is updated using a clipped surrogate objective
(similar to PPO) with these group-relative advantages.

References:
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning
  in Open Language Models" (2024) - introduces GRPO
"""

import copy
from collections import namedtuple
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# Trajectory step storage
TrajectoryStep = namedtuple('TrajectoryStep', [
    'state', 'action', 'log_prob', 'advantage', 'value', 'return_', 'reward', 'done'
])


if TORCH_AVAILABLE:

    class ResBlock(nn.Module):
        """
        Residual block with two convolutions, batch normalization, and a skip
        connection. If the input and output channel counts differ, a 1x1
        convolution is used on the skip path to match dimensions.
        """

        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)

            # 1x1 projection shortcut when channel dimensions change
            if in_channels != out_channels:
                self.skip = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1,
                              stride=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                )
            else:
                self.skip = nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            identity = self.skip(x)
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = out + identity
            return F.relu(out)

    class GRPONetwork(nn.Module):
        """
        Deep ResNet-style policy-value network for GRPO.

        Architecture:
            Conv(3->64, k=3, s=2, p=1) -> BN -> ReLU -> ResBlock(64)
            Conv(64->128, k=3, s=2, p=1) -> BN -> ReLU -> ResBlock(128)
            Conv(128->256, k=3, s=2, p=1) -> BN -> ReLU -> ResBlock(256)
            AdaptiveAvgPool2d(1)
            Policy head: FC(256->256) -> ReLU -> FC(256->num_actions) -> Softmax
            Value head:  FC(256->256) -> ReLU -> FC(256->1)

        The value head provides a state-value baseline for bootstrapping in the
        1-step lookahead used by GRPO, though the primary advantage computation
        is group-relative.
        """

        def __init__(
            self,
            input_channels: int = 3,
            input_size: Tuple[int, int] = (84, 84),
            num_actions: int = 4,
            base_channels: int = 16,
            hidden_dim: int = 64,
        ):
            super().__init__()

            self.input_channels = input_channels
            self.input_size = input_size
            self.num_actions = num_actions
            self.base_channels = base_channels
            self.hidden_dim = hidden_dim
            c1 = base_channels
            c2 = base_channels * 2
            c3 = base_channels * 4

            # --- Encoder backbone ---
            self.stem1 = nn.Sequential(
                nn.Conv2d(input_channels, c1, kernel_size=3, stride=2, padding=1,
                          bias=False),
                nn.BatchNorm2d(c1),
                nn.ReLU(inplace=True),
            )
            self.res1 = ResBlock(c1, c1)

            self.stem2 = nn.Sequential(
                nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1,
                          bias=False),
                nn.BatchNorm2d(c2),
                nn.ReLU(inplace=True),
            )
            self.res2 = ResBlock(c2, c2)

            self.stem3 = nn.Sequential(
                nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1,
                          bias=False),
                nn.BatchNorm2d(c3),
                nn.ReLU(inplace=True),
            )
            self.res3 = ResBlock(c3, c3)

            self.global_pool = nn.AdaptiveAvgPool2d(1)

            # --- Policy head ---
            self.policy_fc1 = nn.Linear(c3, hidden_dim)
            self.policy_fc2 = nn.Linear(hidden_dim, num_actions)

            # --- Value head ---
            self.value_fc1 = nn.Linear(c3, hidden_dim)
            self.value_fc2 = nn.Linear(hidden_dim, 1)

            self._initialize_weights()

        def _initialize_weights(self) -> None:
            """He initialization for conv/linear layers, constant for BN."""
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out',
                                            nonlinearity='relu')
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                    nn.init.constant_(module.bias, 0)

        def _encode(self, x: torch.Tensor) -> torch.Tensor:
            """Run the convolutional backbone and return a flat feature vector."""
            if x.dtype != torch.float32:
                x = x.float()
            if x.max() > 1.0:
                x = x / 255.0

            x = self.stem1(x)
            x = self.res1(x)
            x = self.stem2(x)
            x = self.res2(x)
            x = self.stem3(x)
            x = self.res3(x)
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)  # (B, 256)
            return x

        def forward(
            self, x: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Forward pass returning action probabilities and state value.

            Args:
                x: Observation tensor of shape (B, C, H, W).

            Returns:
                action_probs: Softmax probabilities, shape (B, num_actions).
                value: Scalar state value, shape (B, 1).
            """
            features = self._encode(x)

            # Policy
            policy = F.relu(self.policy_fc1(features))
            action_probs = F.softmax(self.policy_fc2(policy), dim=-1)

            # Value
            value = F.relu(self.value_fc1(features))
            value = self.value_fc2(value)

            return action_probs, value

        def get_parameter_count(self) -> int:
            """Return total number of trainable parameters."""
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

    class GRPOAgent:
        """
        Group Relative Policy Optimization agent.

        At each environment step the agent:
        1. Samples a *group* of actions from the current policy.
        2. Evaluates each action via 1-step lookahead: r + gamma * V(s') if
           not done, else r alone.
        3. Computes group-relative advantages by normalizing within the group.
        4. Stores the transition with its group-relative advantage.

        The policy update uses a clipped surrogate objective (PPO-style) with
        an entropy bonus to encourage exploration and a value-function loss for
        the bootstrap baseline.
        """

        def __init__(
            self,
            input_channels: int = 3,
            input_size: Tuple[int, int] = (84, 84),
            num_actions: int = 4,
            lr: float = 3e-4,
            gamma: float = 0.99,
            group_size: int = 8,
            clip_epsilon: float = 0.2,
            entropy_coef: float = 0.01,
            value_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            gae_lambda: float = 0.95,
            group_advantage_coef: float = 0.5,
            base_channels: int = 16,
            hidden_dim: int = 64,
            device: str = "cpu",
        ):
            """
            Args:
                input_channels: Observation channel count (3 for RGB).
                input_size: Spatial observation size (H, W).
                num_actions: Size of the discrete action space.
                lr: Learning rate for Adam optimizer.
                gamma: Discount factor.
                group_size: Number of actions sampled per state for GRPO.
                clip_epsilon: PPO clipping parameter.
                entropy_coef: Weight of the entropy bonus in the loss.
                value_coef: Weight of the value-function loss.
                max_grad_norm: Maximum gradient norm for clipping.
                gae_lambda: GAE lambda for return computation.
                group_advantage_coef: Weight for one-step group advantage
                    shaping added to the policy advantage.
                base_channels: Base channel width for the shared CNN backbone.
                hidden_dim: Hidden width for policy and value heads.
                device: Torch device string.
            """
            self.device = torch.device(device)
            self.num_actions = num_actions
            self.gamma = gamma
            self.group_size = group_size
            self.clip_epsilon = clip_epsilon
            self.entropy_coef = entropy_coef
            self.value_coef = value_coef
            self.max_grad_norm = max_grad_norm
            self.gae_lambda = gae_lambda
            self.group_advantage_coef = group_advantage_coef

            # Network and optimizer
            self.network = GRPONetwork(
                input_channels=input_channels,
                input_size=input_size,
                num_actions=num_actions,
                base_channels=base_channels,
                hidden_dim=hidden_dim,
            ).to(self.device)

            self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

            # Training bookkeeping
            self.step_count = 0

        # ------------------------------------------------------------------
        # Observation helpers
        # ------------------------------------------------------------------

        def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
            """Convert a single (H, W, C) uint8 observation to (1, C, H, W) tensor."""
            t = torch.from_numpy(obs).float().to(self.device)
            if t.dim() == 3:
                t = t.permute(2, 0, 1)  # HWC -> CHW
            t = t.unsqueeze(0)  # add batch dim
            return t

        @contextmanager
        def _inference_context(self):
            """
            Run inference without updating BatchNorm statistics.

            GRPO rollout collection does many single-state forward passes.
            Leaving the network in training mode for those calls corrupts the
            running statistics and destabilizes the policy.
            """
            was_training = self.network.training
            self.network.eval()
            try:
                with torch.inference_mode():
                    yield
            finally:
                if was_training:
                    self.network.train()

        def _snapshot_env(self, env) -> Dict[str, Any]:
            """Capture enough environment state to undo a simulated step."""
            snapshot: Dict[str, Any] = {
                "engine_state": copy.deepcopy(env.engine._state),
                "engine_rng": copy.deepcopy(env.engine._rng),
            }

            for attr in (
                "_episode_steps",
                "_frame_buffer",
                "_last_score",
                "_prev_distance",
                "_prev_reachable_ratio",
                "_recent_state_signatures",
            ):
                if hasattr(env, attr):
                    snapshot[attr] = copy.deepcopy(getattr(env, attr))

            return snapshot

        def _restore_env(self, env, snapshot: Dict[str, Any]) -> None:
            """Restore the environment after lookahead simulation."""
            env.engine._state = copy.deepcopy(snapshot["engine_state"])
            env.engine._rng = copy.deepcopy(snapshot["engine_rng"])

            if hasattr(env, "_state"):
                env._state = env.engine._state

            for attr in (
                "_episode_steps",
                "_frame_buffer",
                "_last_score",
                "_prev_distance",
                "_prev_reachable_ratio",
                "_recent_state_signatures",
            ):
                if attr in snapshot:
                    setattr(env, attr, copy.deepcopy(snapshot[attr]))

        # ------------------------------------------------------------------
        # Action selection
        # ------------------------------------------------------------------

        def act(self, state: np.ndarray, training: bool = True) -> int:
            """
            Select an action given a raw observation.

            During training the action is sampled from the policy distribution;
            during evaluation the greedy (argmax) action is returned.

            Args:
                state: Raw observation array (H, W, C) uint8.
                training: Whether to sample or take argmax.

            Returns:
                Integer action.
            """
            with self._inference_context():
                state_tensor = self._obs_to_tensor(state)
                action_probs, _ = self.network(state_tensor)
                action_probs = action_probs.squeeze(0)

                if training:
                    dist = Categorical(action_probs)
                    action = dist.sample().item()
                else:
                    action = action_probs.argmax().item()

            return action

        # ------------------------------------------------------------------
        # Group rollouts (core GRPO mechanism)
        # ------------------------------------------------------------------

        def collect_group_rollouts(
            self,
            env,
            obs: np.ndarray,
            num_steps: int = 1,
        ) -> Tuple[np.ndarray, np.ndarray]:
            """
            For the current environment state, simulate each of ``group_size``
            actions via 1-step lookahead and return the estimated values.

            The environment's internal engine state is saved before each
            simulated action and restored afterwards so the real trajectory
            is not affected.

            Args:
                env: A ``SimpleSnakeEnv`` instance (must expose ``.engine``).
                obs: Current observation array (H, W, C).
                num_steps: Number of lookahead steps (only 1 is used).

            Returns:
                group_actions: int array with the evaluated actions.
                group_values: float array with the estimated return for each
                    evaluated action.
            """
            del num_steps  # Current implementation only uses 1-step lookahead.

            # The action space is tiny, so evaluating distinct actions gives a
            # far cleaner relative signal than sampling duplicate actions.
            if self.group_size >= self.num_actions:
                group_actions = np.arange(self.num_actions, dtype=np.int64)
            else:
                with self._inference_context():
                    state_tensor = self._obs_to_tensor(obs)
                    action_probs, _ = self.network(state_tensor)
                    dist = Categorical(action_probs.squeeze(0))
                    sampled = dist.sample((self.group_size * 2,)).cpu().numpy()

                unique_actions: List[int] = []
                for action in sampled.tolist():
                    if action not in unique_actions:
                        unique_actions.append(int(action))
                    if len(unique_actions) == self.group_size:
                        break

                if len(unique_actions) < self.group_size:
                    for action in range(self.num_actions):
                        if action not in unique_actions:
                            unique_actions.append(action)
                        if len(unique_actions) == self.group_size:
                            break

                group_actions = np.asarray(unique_actions, dtype=np.int64)

            saved_env = self._snapshot_env(env)
            group_values = np.zeros(len(group_actions), dtype=np.float32)

            for i, action in enumerate(group_actions):
                self._restore_env(env, saved_env)

                next_obs, reward, terminated, truncated, _info = env.step(
                    int(action)
                )
                done = terminated or truncated

                if done:
                    group_values[i] = reward
                else:
                    with self._inference_context():
                        next_tensor = self._obs_to_tensor(next_obs)
                        _, next_value = self.network(next_tensor)
                        group_values[i] = (
                            reward + self.gamma * next_value.item()
                        )

            self._restore_env(env, saved_env)

            return group_actions, group_values

        # ------------------------------------------------------------------
        # Advantage computation
        # ------------------------------------------------------------------

        @staticmethod
        def compute_group_advantages(
            group_values: np.ndarray,
        ) -> np.ndarray:
            """
            Compute advantages relative to the group mean and standard
            deviation.

            Args:
                group_values: Array with estimated
                    returns for each sampled action.

            Returns:
                Normalized advantages of the same shape.
            """
            mean = group_values.mean()
            std = group_values.std()
            return (group_values - mean) / (std + 1e-8)

        # ------------------------------------------------------------------
        # Trajectory collection
        # ------------------------------------------------------------------

        def collect_trajectory(
            self,
            env,
            max_steps: int = 500,
        ) -> List[TrajectoryStep]:
            """
            Collect one full episode trajectory using GRPO.

            At every timestep:
            1. Sample a group of actions and simulate each via 1-step
               lookahead.
            2. Compute group-relative advantages.
            3. Sample the actual action from the policy (independently of the
               group).
            4. Store the transition with its group-relative advantage.

            After the episode ends, advantages are further refined with GAE
            using the per-step rewards and value estimates collected along
            the way.

            Args:
                env: A ``SimpleSnakeEnv`` instance.
                max_steps: Maximum number of environment steps per episode.

            Returns:
                List of ``TrajectoryStep`` named tuples.
            """
            trajectory: List[TrajectoryStep] = []
            obs, _info = env.reset()

            for _t in range(max_steps):
                # --- Group rollout ---
                group_actions, group_values = self.collect_group_rollouts(
                    env, obs
                )
                group_advantages = self.compute_group_advantages(group_values)

                # --- Policy action selection ---
                state_tensor = self._obs_to_tensor(obs)
                with self._inference_context():
                    action_probs, value = self.network(state_tensor)
                    dist = Categorical(action_probs.squeeze(0))
                    action_t = dist.sample()
                    log_prob = dist.log_prob(action_t)

                action = action_t.item()

                # Find the advantage for the chosen action.  If the action
                # appears among the group samples we take the mean of its
                # advantages; otherwise we fall back to the overall group mean
                # (which is 0 after normalization).
                mask = group_actions == action
                if mask.any():
                    advantage = float(group_advantages[mask].mean())
                else:
                    advantage = 0.0

                # --- Environment step ---
                next_obs, reward, terminated, truncated, _info = env.step(
                    action
                )
                done = terminated or truncated
                self.step_count += 1

                trajectory.append(TrajectoryStep(
                    state=obs,
                    action=action,
                    log_prob=log_prob.item(),
                    advantage=advantage,
                    value=value.item(),
                    return_=0.0,
                    reward=reward,
                    done=done,
                ))

                if done:
                    break

                obs = next_obs

            final_value = 0.0
            if trajectory and not trajectory[-1].done:
                with self._inference_context():
                    final_tensor = self._obs_to_tensor(obs)
                    _, final_value_t = self.network(final_tensor)
                    final_value = final_value_t.item()

            # --- Refine advantages with GAE ---
            trajectory = self._apply_gae(trajectory, final_value)

            return trajectory

        def _apply_gae(
            self,
            trajectory: List[TrajectoryStep],
            final_value: float = 0.0,
        ) -> List[TrajectoryStep]:
            """
            Re-compute advantages using Generalized Advantage Estimation
            (GAE-lambda) blended with the group-relative advantages.

            The final policy advantage adds one-step group-relative shaping to
            the temporal-difference advantage from GAE, while the value target
            remains the actual discounted return.

            Args:
                trajectory: Original trajectory with group advantages.

            Returns:
                New trajectory list with blended advantages and GAE returns.
            """
            n = len(trajectory)
            if n == 0:
                return trajectory

            policy_advantages = np.zeros(n, dtype=np.float32)
            returns = np.zeros(n, dtype=np.float32)
            gae = 0.0
            next_value = final_value

            for t in reversed(range(n)):
                delta = (
                    trajectory[t].reward
                    + self.gamma * next_value * (1.0 - float(trajectory[t].done))
                    - trajectory[t].value
                )
                gae = delta + self.gamma * self.gae_lambda * (
                    1.0 - float(trajectory[t].done)
                ) * gae
                policy_advantages[t] = (
                    gae + self.group_advantage_coef * trajectory[t].advantage
                )
                returns[t] = gae + trajectory[t].value
                next_value = trajectory[t].value

            refined: List[TrajectoryStep] = []
            for t in range(n):
                refined.append(trajectory[t]._replace(
                    advantage=float(policy_advantages[t]),
                    return_=float(returns[t]),
                ))

            return refined

        # ------------------------------------------------------------------
        # Policy update
        # ------------------------------------------------------------------

        def update(
            self,
            trajectories: List[List[TrajectoryStep]],
            epochs: int = 4,
            mini_batch_size: int = 64,
        ) -> Dict[str, float]:
            """
            PPO-style clipped surrogate update using group-relative advantages.

            The method iterates over the collected trajectories for ``epochs``
            passes, sampling mini-batches and applying:
            - Clipped surrogate policy loss.
            - Value function MSE loss.
            - Entropy bonus.

            Args:
                trajectories: List of trajectories (each a list of
                    ``TrajectoryStep``).
                epochs: Number of passes over the data.
                mini_batch_size: Mini-batch size for SGD updates.

            Returns:
                Dictionary of aggregated loss metrics.
            """
            # Flatten all trajectories into arrays
            all_states: List[np.ndarray] = []
            all_actions: List[int] = []
            all_log_probs: List[float] = []
            all_advantages: List[float] = []
            all_returns: List[float] = []

            for traj in trajectories:
                for step in traj:
                    all_states.append(step.state)
                    all_actions.append(step.action)
                    all_log_probs.append(step.log_prob)
                    all_advantages.append(step.advantage)
                    all_returns.append(step.return_)

            n = len(all_states)
            if n == 0:
                return {}

            # Convert to tensors
            states_t = torch.stack(
                [self._obs_to_tensor(s).squeeze(0) for s in all_states]
            ).to(self.device)
            actions_t = torch.tensor(
                all_actions, dtype=torch.long, device=self.device
            )
            old_log_probs_t = torch.tensor(
                all_log_probs, dtype=torch.float32, device=self.device
            )
            advantages_t = torch.tensor(
                all_advantages, dtype=torch.float32, device=self.device
            )
            returns_t = torch.tensor(
                all_returns, dtype=torch.float32, device=self.device
            )

            # Normalize advantages across the full batch
            advantages_t = (
                (advantages_t - advantages_t.mean())
                / (advantages_t.std() + 1e-8)
            )

            # Tracking
            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_entropy = 0.0
            total_loss = 0.0
            num_updates = 0

            indices = np.arange(n)
            self.network.train()

            for _epoch in range(epochs):
                np.random.shuffle(indices)

                for start in range(0, n, mini_batch_size):
                    end = min(start + mini_batch_size, n)
                    mb_idx = indices[start:end]
                    mb_idx_t = torch.from_numpy(mb_idx).long().to(self.device)

                    mb_states = states_t[mb_idx_t]
                    mb_actions = actions_t[mb_idx_t]
                    mb_old_log_probs = old_log_probs_t[mb_idx_t]
                    mb_advantages = advantages_t[mb_idx_t]
                    mb_returns = returns_t[mb_idx_t]

                    # Forward pass
                    action_probs, values = self.network(mb_states)
                    dist = Categorical(action_probs)
                    new_log_probs = dist.log_prob(mb_actions)
                    entropy = dist.entropy().mean()

                    # --- Clipped surrogate policy loss ---
                    ratio = torch.exp(new_log_probs - mb_old_log_probs)
                    surr1 = ratio * mb_advantages
                    surr2 = (
                        torch.clamp(
                            ratio,
                            1.0 - self.clip_epsilon,
                            1.0 + self.clip_epsilon,
                        )
                        * mb_advantages
                    )
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # --- Value loss ---
                    value_loss = F.mse_loss(
                        values.squeeze(-1), mb_returns
                    )

                    # --- Total loss ---
                    loss = (
                        policy_loss
                        + self.value_coef * value_loss
                        - self.entropy_coef * entropy
                    )

                    # Optimisation step
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.network.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()

                    # Accumulate metrics
                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy += entropy.item()
                    total_loss += loss.item()
                    num_updates += 1

            if num_updates == 0:
                return {}

            return {
                "loss": total_loss / num_updates,
                "policy_loss": total_policy_loss / num_updates,
                "value_loss": total_value_loss / num_updates,
                "entropy": total_entropy / num_updates,
                "num_updates": num_updates,
            }

        # ------------------------------------------------------------------
        # Checkpointing
        # ------------------------------------------------------------------

        def save(self, filepath: str) -> None:
            """Save the agent state to a checkpoint file."""
            torch.save({
                "network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step_count": self.step_count,
            }, filepath)

        def load(self, filepath: str) -> None:
            """Load the agent state from a checkpoint file."""
            checkpoint = torch.load(filepath, map_location=self.device)
            self.network.load_state_dict(checkpoint["network"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.step_count = checkpoint["step_count"]

        # ------------------------------------------------------------------
        # Mode switching
        # ------------------------------------------------------------------

        def set_eval_mode(self) -> None:
            """Switch network to evaluation mode (disables dropout / BN updates)."""
            self.network.eval()

        def set_train_mode(self) -> None:
            """Switch network to training mode."""
            self.network.train()


# ---------------------------------------------------------------------------
# Fallback stubs when PyTorch is unavailable
# ---------------------------------------------------------------------------

if not TORCH_AVAILABLE:
    class GRPONetwork:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for GRPONetwork")

    class GRPOAgent:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for GRPOAgent")
