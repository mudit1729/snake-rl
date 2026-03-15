"""
Deep Q-Network (DQN) Implementation

Modern DQN implementation with experience replay, target networks,
double DQN, and configurable exploration strategies.
"""

import random
import math
from collections import deque, namedtuple
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .models.cnn_encoder import create_encoder


# Experience tuple for replay buffer
Experience = namedtuple('Experience', [
    'state', 'action', 'reward', 'next_state', 'done'
])


class ReplayBuffer:
    """
    Experience replay buffer for DQN training.
    
    Stores transitions and provides random sampling for training.
    Implements efficient circular buffer to handle memory constraints.
    """
    
    def __init__(self, capacity: int, device: str = "cpu"):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
            device: Device to store tensors on
        """
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)
        self.position = 0
    
    def push(
        self, 
        state: np.ndarray, 
        action: int, 
        reward: float, 
        next_state: Optional[np.ndarray], 
        done: bool
    ):
        """Add experience to buffer."""
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """
        Sample random batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of tensors if PyTorch available, else raises ImportError
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for replay buffer sampling")
        
        experiences = random.sample(self.buffer, batch_size)
        
        states = torch.stack([
            torch.from_numpy(e.state) for e in experiences
        ]).to(self.device)
        
        actions = torch.tensor([
            e.action for e in experiences
        ], dtype=torch.long, device=self.device)
        
        rewards = torch.tensor([
            e.reward for e in experiences
        ], dtype=torch.float32, device=self.device)
        
        next_states = torch.stack([
            torch.from_numpy(e.next_state) if e.next_state is not None 
            else torch.zeros_like(torch.from_numpy(e.state))
            for e in experiences
        ]).to(self.device)
        
        # Convert from (B, H, W, C) to (B, C, H, W) for PyTorch
        if len(states.shape) == 4:
            states = states.permute(0, 3, 1, 2)
            next_states = next_states.permute(0, 3, 1, 2)
        
        dones = torch.tensor([
            e.done for e in experiences
        ], dtype=torch.bool, device=self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        return len(self.buffer)


class EpsilonScheduler:
    """
    Epsilon-greedy exploration scheduler.
    
    Supports different decay strategies for exploration.
    """
    
    def __init__(
        self,
        initial_epsilon: float = 1.0,
        final_epsilon: float = 0.01,
        decay_steps: int = 100000,
        decay_type: str = "linear"
    ):
        """
        Initialize epsilon scheduler.
        
        Args:
            initial_epsilon: Starting exploration rate
            final_epsilon: Final exploration rate
            decay_steps: Number of steps to decay over
            decay_type: Decay strategy ('linear', 'exponential', 'cosine')
        """
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.decay_steps = decay_steps
        self.decay_type = decay_type
        self.step_count = 0
    
    def get_epsilon(self) -> float:
        """Get current epsilon value."""
        if self.step_count >= self.decay_steps:
            return self.final_epsilon
        
        progress = self.step_count / self.decay_steps
        
        if self.decay_type == "linear":
            epsilon = self.initial_epsilon - progress * (
                self.initial_epsilon - self.final_epsilon
            )
        elif self.decay_type == "exponential":
            decay_rate = math.log(self.final_epsilon / self.initial_epsilon) / self.decay_steps
            epsilon = self.initial_epsilon * math.exp(decay_rate * self.step_count)
        elif self.decay_type == "cosine":
            epsilon = self.final_epsilon + 0.5 * (
                self.initial_epsilon - self.final_epsilon
            ) * (1 + math.cos(math.pi * progress))
        else:
            raise ValueError(f"Unknown decay type: {self.decay_type}")
        
        return max(epsilon, self.final_epsilon)
    
    def step(self):
        """Increment step counter."""
        self.step_count += 1


if TORCH_AVAILABLE:
    class DQNNetwork(nn.Module):
        """
        Deep Q-Network with CNN encoder and value head.
        
        Combines visual feature extraction with Q-value estimation.
        """
        
        def __init__(
            self,
            input_channels: int,
            input_size: Tuple[int, int],
            num_actions: int,
            encoder_type: str = "cnn",
            hidden_dim: int = 512,
            dueling: bool = False,
            **encoder_kwargs
        ):
            """
            Initialize DQN network.
            
            Args:
                input_channels: Number of input channels
                input_size: Input image size (height, width)
                num_actions: Number of possible actions
                encoder_type: Type of encoder ('cnn', 'small', 'small_spatial', 'resnet', 'nature')
                hidden_dim: Hidden layer dimension
                dueling: Whether to use dueling architecture
                **encoder_kwargs: Additional encoder arguments
            """
            super().__init__()
            
            self.num_actions = num_actions
            self.dueling = dueling
            
            # Feature encoder
            self.encoder = create_encoder(
                encoder_type=encoder_type,
                input_channels=input_channels,
                input_size=input_size,
                output_dim=hidden_dim,
                **encoder_kwargs
            )
            
            if dueling:
                # Dueling DQN architecture
                self.value_head = nn.Linear(hidden_dim, 1)
                self.advantage_head = nn.Linear(hidden_dim, num_actions)
            else:
                # Standard DQN architecture
                self.q_head = nn.Linear(hidden_dim, num_actions)
            
            self._initialize_heads()
        
        def _initialize_heads(self):
            """Initialize Q-value heads."""
            if self.dueling:
                nn.init.kaiming_normal_(self.value_head.weight)
                nn.init.constant_(self.value_head.bias, 0)
                nn.init.kaiming_normal_(self.advantage_head.weight)
                nn.init.constant_(self.advantage_head.bias, 0)
            else:
                nn.init.kaiming_normal_(self.q_head.weight)
                nn.init.constant_(self.q_head.bias, 0)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass through network.
            
            Args:
                x: Input observations with shape (batch_size, channels, height, width)
                
            Returns:
                Q-values with shape (batch_size, num_actions)
            """
            # Extract features
            features = self.encoder(x)
            
            if self.dueling:
                # Dueling architecture: Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
                value = self.value_head(features)
                advantage = self.advantage_head(features)
                
                # Subtract mean advantage for stability
                advantage_mean = advantage.mean(dim=1, keepdim=True)
                q_values = value + advantage - advantage_mean
                
                return q_values
            else:
                # Standard DQN
                return self.q_head(features)
        
        def get_parameter_count(self) -> int:
            """Get total number of parameters."""
            return sum(p.numel() for p in self.parameters() if p.requires_grad)


    class DQNAgent:
        """
        Deep Q-Network agent with experience replay and target networks.
        
        Implements DQN, Double DQN, and Dueling DQN variants.
        """
        
        def __init__(
            self,
            input_channels: int,
            input_size: Tuple[int, int],
            num_actions: int,
            lr: float = 1e-4,
            gamma: float = 0.99,
            buffer_size: int = 100000,
            batch_size: int = 32,
            target_update_freq: int = 1000,
            initial_epsilon: float = 1.0,
            final_epsilon: float = 0.01,
            epsilon_decay_steps: int = 100000,
            double_dqn: bool = True,
            dueling: bool = False,
            encoder_type: str = "cnn",
            device: str = "auto",
            **network_kwargs
        ):
            """
            Initialize DQN agent.
            
            Args:
                input_channels: Number of input channels
                input_size: Input image size
                num_actions: Number of possible actions
                lr: Learning rate
                gamma: Discount factor
                buffer_size: Replay buffer size
                batch_size: Training batch size
                target_update_freq: Target network update frequency
                initial_epsilon: Initial exploration rate
                final_epsilon: Final exploration rate
                epsilon_decay_steps: Epsilon decay duration
                double_dqn: Whether to use Double DQN
                dueling: Whether to use Dueling DQN
                encoder_type: Type of encoder
                device: Device for computation
                **network_kwargs: Additional network arguments
            """
            # Set device
            if device == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)
            
            print(f"DQN Agent using device: {self.device}")
            
            self.num_actions = num_actions
            self.gamma = gamma
            self.batch_size = batch_size
            self.target_update_freq = target_update_freq
            self.double_dqn = double_dqn
            
            # Networks
            self.q_network = DQNNetwork(
                input_channels=input_channels,
                input_size=input_size,
                num_actions=num_actions,
                encoder_type=encoder_type,
                dueling=dueling,
                **network_kwargs
            ).to(self.device)
            
            self.target_network = DQNNetwork(
                input_channels=input_channels,
                input_size=input_size,
                num_actions=num_actions,
                encoder_type=encoder_type,
                dueling=dueling,
                **network_kwargs
            ).to(self.device)
            
            # Copy weights to target network
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.target_network.eval()
            
            # Optimizer
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
            
            # Replay buffer
            self.replay_buffer = ReplayBuffer(buffer_size, device=self.device)
            
            # Exploration
            self.epsilon_scheduler = EpsilonScheduler(
                initial_epsilon=initial_epsilon,
                final_epsilon=final_epsilon,
                decay_steps=epsilon_decay_steps
            )
            
            # Training state
            self.step_count = 0
            self.update_count = 0
            
            print(f"DQN Network parameters: {self.q_network.get_parameter_count():,}")
        
        def act(self, state: np.ndarray, training: bool = True) -> int:
            """
            Select action using epsilon-greedy policy.
            
            Args:
                state: Current state observation
                training: Whether in training mode
                
            Returns:
                Selected action
            """
            if training and random.random() < self.epsilon_scheduler.get_epsilon():
                return random.randrange(self.num_actions)
            
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
                # Convert from (B, H, W, C) to (B, C, H, W) for PyTorch
                if len(state_tensor.shape) == 4:
                    state_tensor = state_tensor.permute(0, 3, 1, 2)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
        
        def store_experience(
            self,
            state: np.ndarray,
            action: int,
            reward: float,
            next_state: Optional[np.ndarray],
            done: bool
        ):
            """Store experience in replay buffer."""
            self.replay_buffer.push(state, action, reward, next_state, done)
        
        def update(self) -> Dict[str, float]:
            """
            Perform one training update.
            
            Returns:
                Dictionary of training metrics
            """
            if len(self.replay_buffer) < self.batch_size:
                return {}
            
            # Sample batch
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(
                self.batch_size
            )
            
            # Current Q-values
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
            
            # Next Q-values
            with torch.no_grad():
                if self.double_dqn:
                    # Double DQN: use main network to select, target to evaluate
                    next_actions = self.q_network(next_states).argmax(1, keepdim=True)
                    next_q_values = self.target_network(next_states).gather(1, next_actions)
                else:
                    # Standard DQN: use target network for both
                    next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
                
                target_q_values = rewards.unsqueeze(1) + (
                    self.gamma * next_q_values * (~dones).unsqueeze(1)
                )
            
            # Compute loss
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
            
            self.optimizer.step()
            
            # Update target network
            self.update_count += 1
            if self.update_count % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
            
            # Update epsilon
            self.epsilon_scheduler.step()
            
            return {
                "loss": loss.item(),
                "epsilon": self.epsilon_scheduler.get_epsilon(),
                "q_value_mean": current_q_values.mean().item(),
                "target_q_mean": target_q_values.mean().item(),
            }
        
        def save(self, filepath: str):
            """Save agent state."""
            torch.save({
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step_count": self.step_count,
                "update_count": self.update_count,
                "epsilon_scheduler": {
                    "step_count": self.epsilon_scheduler.step_count,
                    "initial_epsilon": self.epsilon_scheduler.initial_epsilon,
                    "final_epsilon": self.epsilon_scheduler.final_epsilon,
                    "decay_steps": self.epsilon_scheduler.decay_steps,
                    "decay_type": self.epsilon_scheduler.decay_type,
                }
            }, filepath)
        
        def load(self, filepath: str):
            """Load agent state."""
            checkpoint = torch.load(filepath, map_location=self.device)
            
            self.q_network.load_state_dict(checkpoint["q_network"])
            self.target_network.load_state_dict(checkpoint["target_network"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.step_count = checkpoint["step_count"]
            self.update_count = checkpoint["update_count"]
            
            # Restore epsilon scheduler
            eps_state = checkpoint["epsilon_scheduler"]
            self.epsilon_scheduler = EpsilonScheduler(
                initial_epsilon=eps_state["initial_epsilon"],
                final_epsilon=eps_state["final_epsilon"],
                decay_steps=eps_state["decay_steps"],
                decay_type=eps_state["decay_type"]
            )
            self.epsilon_scheduler.step_count = eps_state["step_count"]
        
        def set_eval_mode(self):
            """Set networks to evaluation mode."""
            self.q_network.eval()
        
        def set_train_mode(self):
            """Set networks to training mode."""
            self.q_network.train()


def test_dqn_functionality():
    """Test DQN implementation with dummy data."""
    if not TORCH_AVAILABLE:
        print("PyTorch not available for DQN testing")
        return
    
    print("Testing DQN functionality...")
    
    # Create agent
    agent = DQNAgent(
        input_channels=3,
        input_size=(84, 84),
        num_actions=4,
        buffer_size=1000,
        batch_size=32,
        lr=1e-4
    )
    
    print(f"✅ Agent created with {agent.q_network.get_parameter_count():,} parameters")
    
    # Test action selection
    dummy_state = np.random.randint(0, 255, (3, 84, 84), dtype=np.uint8)
    action = agent.act(dummy_state)
    print(f"✅ Action selection: {action}")
    
    # Test experience storage and training
    for i in range(100):
        state = np.random.randint(0, 255, (3, 84, 84), dtype=np.uint8)
        action = agent.act(state)
        reward = random.uniform(-1, 1)
        next_state = np.random.randint(0, 255, (3, 84, 84), dtype=np.uint8)
        done = random.random() < 0.1
        
        agent.store_experience(state, action, reward, next_state, done)
    
    print(f"✅ Stored {len(agent.replay_buffer)} experiences")
    
    # Test training update
    metrics = agent.update()
    if metrics:
        print(f"✅ Training update: loss={metrics.get('loss', 0):.4f}")
    
    print("✅ DQN functionality test passed!")


if __name__ == "__main__":
    test_dqn_functionality()
else:
    # Provide dummy classes if PyTorch is not available
    if not TORCH_AVAILABLE:
        class DQNAgent:
            def __init__(self, *args, **kwargs):
                raise ImportError("PyTorch is required for DQN agent")
        
        class DQNNetwork:
            def __init__(self, *args, **kwargs):
                raise ImportError("PyTorch is required for DQN network")
