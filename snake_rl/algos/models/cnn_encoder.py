"""
CNN Encoder for Snake Game

Atari-style convolutional neural network for processing visual observations
from the Snake environment. Designed for 84x84 RGB inputs with ~1M parameters.
"""

from typing import Tuple, Optional, Union
import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class CNNEncoder(nn.Module):
        """
        Convolutional neural network encoder for Snake game observations.
        
        Architecture based on Atari DQN paper with modifications for Snake:
        - 3 convolutional layers with ReLU activations
        - Adaptive pooling to handle different input sizes
        - Fully connected layer to produce feature embeddings
        - ~1M parameters for 84x84 RGB input
        """
        
        def __init__(
            self,
            input_channels: int = 3,
            input_size: Tuple[int, int] = (84, 84),
            output_dim: int = 512,
            activation: str = "relu",
        ):
            """
            Initialize CNN encoder.
            
            Args:
                input_channels: Number of input channels (3 for RGB, 1 for grayscale)
                input_size: Input image size (height, width)
                output_dim: Output feature dimension
                activation: Activation function ('relu', 'leaky_relu', 'swish')
            """
            super().__init__()
            
            self.input_channels = input_channels
            self.input_size = input_size
            self.output_dim = output_dim
            
            # Choose activation function
            if activation == "relu":
                self.activation = nn.ReLU(inplace=True)
            elif activation == "leaky_relu":
                self.activation = nn.LeakyReLU(0.1, inplace=True)
            elif activation == "swish":
                self.activation = nn.SiLU(inplace=True)
            else:
                raise ValueError(f"Unknown activation: {activation}")
            
            # Convolutional layers (Atari-style)
            self.conv1 = nn.Conv2d(
                in_channels=input_channels,
                out_channels=32,
                kernel_size=8,
                stride=4,
                padding=0
            )
            
            self.conv2 = nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=0
            )
            
            self.conv3 = nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=0
            )
            
            # Calculate the size after convolutions
            self.conv_output_size = self._calculate_conv_output_size()
            
            # Fully connected layer
            self.fc = nn.Linear(self.conv_output_size, output_dim)
            
            # Initialize weights
            self._initialize_weights()
        
        def _calculate_conv_output_size(self) -> int:
            """Calculate the output size of convolutional layers."""
            # Create a dummy input to calculate output size
            with torch.no_grad():
                dummy_input = torch.zeros(1, self.input_channels, *self.input_size)
                x = self.activation(self.conv1(dummy_input))
                x = self.activation(self.conv2(x))
                x = self.activation(self.conv3(x))
                return x.numel()
        
        def _initialize_weights(self):
            """Initialize network weights using He initialization."""
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant_(module.bias, 0)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass through the network.
            
            Args:
                x: Input tensor with shape (batch_size, channels, height, width)
                
            Returns:
                Feature tensor with shape (batch_size, output_dim)
            """
            # Ensure input is float and normalized
            if x.dtype != torch.float32:
                x = x.float()
            
            # Normalize to [0, 1] if input appears to be in [0, 255]
            if x.max() > 1.0:
                x = x / 255.0
            
            # Convolutional layers
            x = self.activation(self.conv1(x))
            x = self.activation(self.conv2(x))
            x = self.activation(self.conv3(x))
            
            # Flatten and fully connected
            x = x.reshape(x.size(0), -1)
            x = self.fc(x)
            
            return x
        
        def get_feature_dim(self) -> int:
            """Get the output feature dimension."""
            return self.output_dim
        
        def get_parameter_count(self) -> int:
            """Get the total number of parameters."""
            return sum(p.numel() for p in self.parameters() if p.requires_grad)


    class SmallCNNEncoder(nn.Module):
        """
        CNN encoder designed for small inputs (e.g. 42x42).

        Uses smaller kernels and strides to preserve spatial information.
        On a 42x42 input this produces an 11x11x64 feature map (vs 1x1x64
        with the Atari-style encoder), retaining spatial structure.
        """

        def __init__(
            self,
            input_channels: int = 3,
            input_size: Tuple[int, int] = (42, 42),
            output_dim: int = 512,
            activation: str = "relu",
        ):
            """
            Initialize small-input CNN encoder.

            Args:
                input_channels: Number of input channels (3 for RGB, 1 for grayscale)
                input_size: Input image size (height, width)
                output_dim: Output feature dimension
                activation: Activation function ('relu', 'leaky_relu', 'swish')
            """
            super().__init__()

            self.input_channels = input_channels
            self.input_size = input_size
            self.output_dim = output_dim

            # Choose activation function
            if activation == "relu":
                self.activation = nn.ReLU(inplace=True)
            elif activation == "leaky_relu":
                self.activation = nn.LeakyReLU(0.1, inplace=True)
            elif activation == "swish":
                self.activation = nn.SiLU(inplace=True)
            else:
                raise ValueError(f"Unknown activation: {activation}")

            # Convolutional layers (small-input friendly)
            # Conv1: input_channels -> 32, kernel=3, stride=2, padding=1 => 21x21
            self.conv1 = nn.Conv2d(
                in_channels=input_channels,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
            )

            # Conv2: 32 -> 64, kernel=3, stride=2, padding=1 => 11x11
            self.conv2 = nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
            )

            # Conv3: 64 -> 64, kernel=3, stride=1, padding=1 => 11x11
            self.conv3 = nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            )

            # Conv4: 64 -> 64, kernel=3, stride=1, padding=1 => 11x11
            self.conv4 = nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            )

            # Calculate the size after convolutions
            self.conv_output_size = self._calculate_conv_output_size()

            # Fully connected layer
            self.fc = nn.Linear(self.conv_output_size, output_dim)

            # Initialize weights
            self._initialize_weights()

        def _calculate_conv_output_size(self) -> int:
            """Calculate the output size of convolutional layers."""
            with torch.no_grad():
                dummy_input = torch.zeros(1, self.input_channels, *self.input_size)
                x = self.activation(self.conv1(dummy_input))
                x = self.activation(self.conv2(x))
                x = self.activation(self.conv3(x))
                x = self.activation(self.conv4(x))
                return x.numel()

        def _initialize_weights(self):
            """Initialize network weights using He initialization."""
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant_(module.bias, 0)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass through the network.

            Args:
                x: Input tensor with shape (batch_size, channels, height, width)

            Returns:
                Feature tensor with shape (batch_size, output_dim)
            """
            # Ensure input is float and normalized
            if x.dtype != torch.float32:
                x = x.float()

            # Normalize to [0, 1] if input appears to be in [0, 255]
            if x.max() > 1.0:
                x = x / 255.0

            # Convolutional layers
            x = self.activation(self.conv1(x))
            x = self.activation(self.conv2(x))
            x = self.activation(self.conv3(x))
            x = self.activation(self.conv4(x))

            # Flatten and fully connected
            x = x.reshape(x.size(0), -1)
            x = self.fc(x)

            return x

        def get_feature_dim(self) -> int:
            """Get the output feature dimension."""
            return self.output_dim

        def get_parameter_count(self) -> int:
            """Get the total number of parameters."""
            return sum(p.numel() for p in self.parameters() if p.requires_grad)


    class SpatialPooledCNNEncoder(nn.Module):
        """
        Compact CNN encoder for small inputs that keeps coarse board layout.

        The original small encoder flattened the full 11x11x64 feature map into
        one large linear layer. This variant keeps the same small-input friendly
        conv stack, then uses a strided downsampling conv before the projection
        head. That removes the giant FC bottleneck while preserving enough
        spatial structure for Snake and stays compatible with MPS.
        """

        def __init__(
            self,
            input_channels: int = 3,
            input_size: Tuple[int, int] = (42, 42),
            output_dim: int = 128,
            activation: str = "relu",
        ):
            super().__init__()

            self.input_channels = input_channels
            self.input_size = input_size
            self.output_dim = output_dim

            if activation == "relu":
                self.activation = nn.ReLU(inplace=True)
            elif activation == "leaky_relu":
                self.activation = nn.LeakyReLU(0.1, inplace=True)
            elif activation == "swish":
                self.activation = nn.SiLU(inplace=True)
            else:
                raise ValueError(f"Unknown activation: {activation}")

            self.conv1 = nn.Conv2d(
                in_channels=input_channels,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
            )
            self.conv2 = nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
            )
            self.conv3 = nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            self.conv4 = nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            self.conv5 = nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
            )

            self.downsampled_output_size = self._calculate_downsampled_output_size()
            self.fc = nn.Linear(self.downsampled_output_size, output_dim)

            self._initialize_weights()

        def _calculate_downsampled_output_size(self) -> int:
            """Calculate the post-downsampling feature size."""
            with torch.no_grad():
                dummy_input = torch.zeros(1, self.input_channels, *self.input_size)
                x = self.activation(self.conv1(dummy_input))
                x = self.activation(self.conv2(x))
                x = self.activation(self.conv3(x))
                x = self.activation(self.conv4(x))
                x = self.activation(self.conv5(x))
                return x.numel()

        def _initialize_weights(self):
            """Initialize network weights using He initialization."""
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant_(module.bias, 0)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dtype != torch.float32:
                x = x.float()

            if x.max() > 1.0:
                x = x / 255.0

            x = self.activation(self.conv1(x))
            x = self.activation(self.conv2(x))
            x = self.activation(self.conv3(x))
            x = self.activation(self.conv4(x))
            x = self.activation(self.conv5(x))
            x = x.reshape(x.size(0), -1)
            x = self.activation(self.fc(x))
            return x

        def get_feature_dim(self) -> int:
            return self.output_dim

        def get_parameter_count(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)


    class ResidualBlock(nn.Module):
        """Residual block for deeper CNN architectures."""
        
        def __init__(self, channels: int, activation: nn.Module):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
            self.activation = activation
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x
            x = self.activation(self.conv1(x))
            x = self.conv2(x)
            x = x + residual
            return self.activation(x)


    class ResNetEncoder(nn.Module):
        """
        ResNet-style encoder for more complex visual features.
        
        Uses residual connections to enable deeper networks while
        maintaining gradient flow.
        """
        
        def __init__(
            self,
            input_channels: int = 3,
            input_size: Tuple[int, int] = (84, 84),
            output_dim: int = 512,
            num_residual_blocks: int = 2,
            activation: str = "relu",
        ):
            super().__init__()
            
            self.input_channels = input_channels
            self.input_size = input_size
            self.output_dim = output_dim
            
            # Choose activation
            if activation == "relu":
                self.activation = nn.ReLU(inplace=True)
            elif activation == "leaky_relu":
                self.activation = nn.LeakyReLU(0.1, inplace=True)
            else:
                self.activation = nn.ReLU(inplace=True)
            
            # Initial convolution
            self.conv1 = nn.Conv2d(input_channels, 32, 8, stride=4, padding=2)
            
            # Residual blocks
            self.res_blocks1 = nn.Sequential(*[
                ResidualBlock(32, self.activation) for _ in range(num_residual_blocks)
            ])
            
            # Downsampling
            self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
            
            self.res_blocks2 = nn.Sequential(*[
                ResidualBlock(64, self.activation) for _ in range(num_residual_blocks)
            ])
            
            # Final convolution
            self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
            
            # Calculate output size
            self.conv_output_size = self._calculate_conv_output_size()
            
            # Global average pooling and FC
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(64, output_dim)
            
            self._initialize_weights()
        
        def _calculate_conv_output_size(self) -> int:
            with torch.no_grad():
                dummy_input = torch.zeros(1, self.input_channels, *self.input_size)
                x = self.activation(self.conv1(dummy_input))
                x = self.res_blocks1(x)
                x = self.activation(self.conv2(x))
                x = self.res_blocks2(x)
                x = self.activation(self.conv3(x))
                return x.numel()
        
        def _initialize_weights(self):
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight)
                    nn.init.constant_(module.bias, 0)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dtype != torch.float32:
                x = x.float()
            
            if x.max() > 1.0:
                x = x / 255.0
            
            # Forward pass
            x = self.activation(self.conv1(x))
            x = self.res_blocks1(x)
            x = self.activation(self.conv2(x))
            x = self.res_blocks2(x)
            x = self.activation(self.conv3(x))
            
            # Global pooling and FC
            x = self.global_pool(x)
            x = x.reshape(x.size(0), -1)
            x = self.fc(x)
            
            return x
        
        def get_feature_dim(self) -> int:
            return self.output_dim
        
        def get_parameter_count(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)


    class NatureEncoder(nn.Module):
        """
        Nature DQN encoder implementation.
        
        Exact architecture from "Human-level control through deep reinforcement
        learning" (Mnih et al., 2015).
        """
        
        def __init__(
            self,
            input_channels: int = 4,  # Frame stacking
            input_size: Tuple[int, int] = (84, 84),
            output_dim: int = 512,
        ):
            super().__init__()
            
            self.input_channels = input_channels
            self.input_size = input_size
            self.output_dim = output_dim
            
            # Nature DQN architecture
            self.conv1 = nn.Conv2d(input_channels, 32, 8, stride=4)
            self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
            self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
            
            # Calculate conv output size
            self.conv_output_size = self._calculate_conv_output_size()
            
            # Fully connected layers
            self.fc1 = nn.Linear(self.conv_output_size, 512)
            self.fc2 = nn.Linear(512, output_dim)
            
            self._initialize_weights()
        
        def _calculate_conv_output_size(self) -> int:
            with torch.no_grad():
                dummy_input = torch.zeros(1, self.input_channels, *self.input_size)
                x = F.relu(self.conv1(dummy_input))
                x = F.relu(self.conv2(x))
                x = F.relu(self.conv3(x))
                return x.numel()
        
        def _initialize_weights(self):
            for module in self.modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dtype != torch.float32:
                x = x.float()
            
            if x.max() > 1.0:
                x = x / 255.0
            
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            
            x = x.reshape(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            
            return x
        
        def get_feature_dim(self) -> int:
            return self.output_dim
        
        def get_parameter_count(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_encoder(
    encoder_type: str = "cnn",
    input_channels: int = 3,
    input_size: Tuple[int, int] = (84, 84),
    output_dim: int = 512,
    **kwargs
) -> Union["CNNEncoder", "SmallCNNEncoder", "SpatialPooledCNNEncoder", "ResNetEncoder", "NatureEncoder"]:
    """
    Factory function to create different encoder types.
    
    Args:
        encoder_type: Type of encoder ('cnn', 'small', 'small_spatial', 'resnet', 'nature')
        input_channels: Number of input channels
        input_size: Input image size (height, width)
        output_dim: Output feature dimension
        **kwargs: Additional arguments for specific encoders
        
    Returns:
        Configured encoder model
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for neural network models")
    
    if encoder_type == "cnn":
        return CNNEncoder(
            input_channels=input_channels,
            input_size=input_size,
            output_dim=output_dim,
            **kwargs
        )
    elif encoder_type == "small":
        return SmallCNNEncoder(
            input_channels=input_channels,
            input_size=input_size,
            output_dim=output_dim,
            **kwargs
        )
    elif encoder_type == "small_spatial":
        return SpatialPooledCNNEncoder(
            input_channels=input_channels,
            input_size=input_size,
            output_dim=output_dim,
            **kwargs
        )
    elif encoder_type == "resnet":
        return ResNetEncoder(
            input_channels=input_channels,
            input_size=input_size,
            output_dim=output_dim,
            **kwargs
        )
    elif encoder_type == "nature":
        return NatureEncoder(
            input_channels=input_channels,
            input_size=input_size,
            output_dim=output_dim,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


def test_encoder_shapes():
    """Test encoder with different input shapes."""
    if not TORCH_AVAILABLE:
        print("PyTorch not available for encoder testing")
        return
    
    print("Testing CNN encoder shapes...")
    
    # Test different configurations
    configs = [
        {"input_channels": 3, "input_size": (84, 84), "name": "RGB 84x84"},
        {"input_channels": 1, "input_size": (64, 64), "name": "Gray 64x64"},
        {"input_channels": 4, "input_size": (84, 84), "name": "Stacked 84x84"},
        {"input_channels": 3, "input_size": (96, 96), "name": "RGB 96x96"},
    ]
    
    for config in configs:
        try:
            encoder = create_encoder(
                encoder_type="cnn",
                input_channels=config["input_channels"],
                input_size=config["input_size"],
                output_dim=512
            )
            
            # Test forward pass
            batch_size = 4
            dummy_input = torch.randn(
                batch_size, 
                config["input_channels"], 
                *config["input_size"]
            )
            
            output = encoder(dummy_input)
            param_count = encoder.get_parameter_count()
            
            print(f"  {config['name']}: "
                  f"input {dummy_input.shape} -> output {output.shape}, "
                  f"params {param_count:,}")
            
        except Exception as e:
            print(f"  {config['name']}: Error - {e}")


if __name__ == "__main__":
    test_encoder_shapes()
else:
    # Provide dummy classes if PyTorch is not available
    if not TORCH_AVAILABLE:
        class CNNEncoder:
            def __init__(self, *args, **kwargs):
                raise ImportError("PyTorch is required for CNN encoder")
        
        class SmallCNNEncoder:
            def __init__(self, *args, **kwargs):
                raise ImportError("PyTorch is required for SmallCNN encoder")

        class SpatialPooledCNNEncoder:
            def __init__(self, *args, **kwargs):
                raise ImportError("PyTorch is required for SpatialPooledCNN encoder")

        class ResNetEncoder:
            def __init__(self, *args, **kwargs):
                raise ImportError("PyTorch is required for ResNet encoder")
        
        class NatureEncoder:
            def __init__(self, *args, **kwargs):
                raise ImportError("PyTorch is required for Nature encoder")
        
        def create_encoder(*args, **kwargs):
            raise ImportError("PyTorch is required for encoders")
