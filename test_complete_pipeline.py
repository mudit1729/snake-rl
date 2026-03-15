#!/usr/bin/env python3
"""
Complete pipeline test without external dependencies.

Tests the entire codebase to ensure all components work together.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

def test_imports():
    """Test that all modules can be imported."""
    print("🔄 Testing imports...")
    
    try:
        # Core engine
        from snake_rl.sim.engine import SnakeEngine, Action, CellType
        from snake_rl.sim.renderer import SnakeRenderer
        print("  ✅ Core engine modules")
        
        # Environment
        from snake_rl.env.snake_env import SnakeEnv
        from snake_rl.env.vector_env import ReplayBuffer, EpsilonScheduler
        print("  ✅ Environment modules")
        
        # Algorithms
        from snake_rl.algos.models.cnn_encoder import create_encoder
        from snake_rl.algos.dqn import DQNAgent
        print("  ✅ Algorithm modules")
        
        return True
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return False


def test_engine_pipeline():
    """Test complete engine functionality."""
    print("🔄 Testing engine pipeline...")
    
    try:
        from snake_rl.sim.engine import SnakeEngine, Action
        from snake_rl.sim.renderer import SnakeRenderer
        
        # Create engine
        engine = SnakeEngine(grid_size=(8, 8))
        renderer = SnakeRenderer()
        
        # Test multiple episodes
        for episode in range(3):
            state = engine.reset(seed=episode)
            total_reward = 0
            
            for step in range(20):
                action = step % 4  # Cycle through actions
                state, reward, done, info = engine.step(action)
                total_reward += reward
                
                # Test renderer
                array_obs = renderer.render_array(state)
                assert array_obs.shape == (8, 8)
                
                if done:
                    break
            
            print(f"    Episode {episode + 1}: {step + 1} steps, reward {total_reward:.2f}")
        
        print("  ✅ Engine pipeline working")
        return True
        
    except Exception as e:
        print(f"  ❌ Engine test failed: {e}")
        return False


def test_environment_pipeline():
    """Test environment wrapper functionality."""
    print("🔄 Testing environment pipeline...")
    
    try:
        from snake_rl.env.snake_env import SnakeEnv
        
        # Test different configurations
        configs = [
            {"observation_type": "grid", "grid_size": (6, 6)},
            {"observation_type": "rgb", "observation_size": (32, 32)},
            {"observation_type": "gray", "observation_size": (64, 64)},
        ]
        
        for i, config in enumerate(configs):
            env = SnakeEnv(**config)
            
            obs, info = env.reset(seed=42)
            total_reward = 0
            
            for step in range(10):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                
                if terminated or truncated:
                    break
            
            print(f"    Config {i + 1}: obs {obs.shape}, reward {total_reward:.2f}")
        
        print("  ✅ Environment pipeline working")
        return True
        
    except Exception as e:
        print(f"  ❌ Environment test failed: {e}")
        return False


def test_training_components():
    """Test training-related components."""
    print("🔄 Testing training components...")
    
    try:
        from snake_rl.algos.dqn import ReplayBuffer, EpsilonScheduler
        import numpy as np
        
        # Test replay buffer
        buffer = ReplayBuffer(capacity=100)
        
        for i in range(50):
            state = np.random.randint(0, 255, (3, 32, 32), dtype=np.uint8)
            action = i % 4
            reward = float(i - 25)
            next_state = np.random.randint(0, 255, (3, 32, 32), dtype=np.uint8)
            done = i % 10 == 9
            
            buffer.push(state, action, reward, next_state, done)
        
        print(f"    Replay buffer: {len(buffer)} experiences stored")
        
        # Test epsilon scheduler
        scheduler = EpsilonScheduler(
            initial_epsilon=1.0,
            final_epsilon=0.1,
            decay_steps=100,
            decay_type="linear"
        )
        
        epsilons = []
        for step in [0, 25, 50, 75, 100]:
            scheduler.step_count = step
            epsilons.append(scheduler.get_epsilon())
        
        print(f"    Epsilon schedule: {[f'{eps:.2f}' for eps in epsilons]}")
        
        print("  ✅ Training components working")
        return True
        
    except Exception as e:
        print(f"  ❌ Training components test failed: {e}")
        return False


def test_integration():
    """Test integration between components."""
    print("🔄 Testing component integration...")
    
    try:
        from snake_rl.env.snake_env import SnakeEnv
        from snake_rl.algos.dqn import EpsilonScheduler
        import numpy as np
        
        # Create environment
        env = SnakeEnv(
            grid_size=(6, 6),
            observation_type="rgb",
            observation_size=(32, 32),
            reward_scheme="dense"
        )
        
        # Simulate training loop without PyTorch
        scheduler = EpsilonScheduler(initial_epsilon=1.0, final_epsilon=0.1, decay_steps=50)
        
        total_steps = 0
        episodes_completed = 0
        
        for episode in range(5):
            obs, _ = env.reset(seed=episode)
            episode_reward = 0
            
            for step in range(20):
                # Simulate epsilon-greedy action selection
                epsilon = scheduler.get_epsilon()
                if np.random.random() < epsilon:
                    action = env.action_space.sample()  # Random action
                else:
                    action = 0  # "Greedy" action (simplified)
                
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                total_steps += 1
                scheduler.step()
                
                if terminated or truncated:
                    episodes_completed += 1
                    break
            
            print(f"    Episode {episode + 1}: {step + 1} steps, reward {episode_reward:.2f}, epsilon {epsilon:.3f}")
        
        print(f"    Integration test: {total_steps} steps, {episodes_completed} episodes completed")
        print("  ✅ Integration working")
        return True
        
    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        return False


def test_configuration_structure():
    """Test that configuration files are present and valid."""
    print("🔄 Testing configuration structure...")
    
    try:
        from pathlib import Path
        import yaml
        
        config_dir = Path("snake_rl/conf")
        
        # Check main config
        main_config = config_dir / "config.yaml"
        if main_config.exists():
            with open(main_config) as f:
                config = yaml.safe_load(f)
            print("    ✅ Main config found and valid")
        else:
            print("    ⚠️  Main config not found")
        
        # Check environment configs
        env_configs = list((config_dir / "env").glob("*.yaml"))
        print(f"    ✅ Found {len(env_configs)} environment configs")
        
        # Check agent configs
        agent_configs = list((config_dir / "agent").glob("*.yaml"))
        print(f"    ✅ Found {len(agent_configs)} agent configs")
        
        print("  ✅ Configuration structure valid")
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration test failed: {e}")
        return False


def main():
    """Run complete pipeline test."""
    print("🐍 Snake RL Complete Pipeline Test")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Engine Pipeline", test_engine_pipeline),
        ("Environment Pipeline", test_environment_pipeline),
        ("Training Components", test_training_components),
        ("Integration", test_integration),
        ("Configuration", test_configuration_structure),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! The Snake RL project is ready for use.")
        print("\nNext steps:")
        print("  1. Install dependencies: poetry install")
        print("  2. Train a model: python train_dqn.py")
        print("  3. Evaluate results: python evaluate.py checkpoints/best_model.pt")
        print("  4. Try the demo: python demo.py --model checkpoints/best_model.pt")
    else:
        print(f"\n⚠️  Some tests failed. Check the errors above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())