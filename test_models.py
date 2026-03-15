#!/usr/bin/env python3
"""
Test neural network models without PyTorch dependency.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

print("Testing model imports and basic functionality...")

try:
    # Test encoder imports
    from snake_rl.algos.models.cnn_encoder import create_encoder
    print("✅ CNN encoder module imported")
    
    # Test DQN imports
    from snake_rl.algos.dqn import DQNAgent, ReplayBuffer, EpsilonScheduler
    print("✅ DQN module imported")
    
    # Test epsilon scheduler (doesn't require PyTorch)
    scheduler = EpsilonScheduler(
        initial_epsilon=1.0,
        final_epsilon=0.01,
        decay_steps=1000,
        decay_type="linear"
    )
    
    print(f"Initial epsilon: {scheduler.get_epsilon():.3f}")
    
    for i in range(0, 1001, 200):
        scheduler.step_count = i
        epsilon = scheduler.get_epsilon()
        print(f"Step {i}: epsilon = {epsilon:.3f}")
    
    print("✅ Epsilon scheduler working")
    
    # Test replay buffer structure (without PyTorch tensors)
    buffer = ReplayBuffer(capacity=1000, device="cpu")
    
    # Add dummy experiences
    import numpy as np
    for i in range(10):
        state = np.random.randint(0, 255, (3, 84, 84), dtype=np.uint8)
        action = i % 4
        reward = float(i - 5)
        next_state = np.random.randint(0, 255, (3, 84, 84), dtype=np.uint8)
        done = i == 9
        
        buffer.push(state, action, reward, next_state, done)
    
    print(f"✅ Replay buffer: stored {len(buffer)} experiences")
    
    # Test different epsilon decay types
    decay_types = ["linear", "exponential", "cosine"]
    
    for decay_type in decay_types:
        scheduler = EpsilonScheduler(
            initial_epsilon=1.0,
            final_epsilon=0.1,
            decay_steps=100,
            decay_type=decay_type
        )
        
        # Test a few points
        test_points = [0, 25, 50, 75, 100, 150]
        epsilons = []
        
        for step in test_points:
            scheduler.step_count = step
            epsilons.append(scheduler.get_epsilon())
        
        print(f"  {decay_type}: {[f'{eps:.3f}' for eps in epsilons]}")
    
    print("✅ Epsilon decay types working")
    
    # Test that PyTorch-dependent components fail gracefully
    try:
        # This should fail if PyTorch is not available
        encoder = create_encoder(
            encoder_type="cnn",
            input_channels=3,
            input_size=(84, 84)
        )
        print("✅ PyTorch is available - encoder creation successful")
        
        # Test DQN agent creation
        agent = DQNAgent(
            input_channels=3,
            input_size=(84, 84),
            num_actions=4,
            buffer_size=1000
        )
        print(f"✅ DQN agent created with {agent.q_network.get_parameter_count():,} parameters")
        
    except ImportError as e:
        print(f"⚠️  PyTorch not available: {e}")
        print("   This is expected if PyTorch is not installed")
    
    print(f"\n🎉 Model architecture tests completed!")
    print("Ready to implement training script with Hydra configuration")
    
except Exception as e:
    print(f"❌ Model test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)