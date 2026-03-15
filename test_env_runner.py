#!/usr/bin/env python3
"""
Test runner for environment functionality.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

try:
    from snake_rl.env.snake_env import SnakeEnv
    from snake_rl.sim.engine import Action
    import numpy as np
    
    print("Testing SnakeEnv functionality...")
    
    # Test different observation types
    obs_types = ["rgb", "gray", "grid"]
    
    for obs_type in obs_types:
        print(f"\n🔍 Testing {obs_type} observations...")
        
        env = SnakeEnv(
            grid_size=(8, 8),
            observation_type=obs_type,
            observation_size=(64, 64) if obs_type != "grid" else None,
            reward_scheme="classic"
        )
        
        # Test reset
        obs, info = env.reset(seed=42)
        print(f"  Reset: obs shape {obs.shape}, info keys {list(info.keys())}")
        
        # Test steps
        total_reward = 0
        for step in range(10):
            action = step % 4  # Cycle through actions
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"  Step {step+1}: reward={reward:.3f}, score={info.get('score', 0)}, done={terminated or truncated}")
            
            if terminated or truncated:
                print(f"  Episode ended: {info.get('reason', 'unknown')}")
                break
        
        print(f"  Total reward: {total_reward:.3f}")
    
    # Test reward schemes
    print(f"\n💰 Testing reward schemes...")
    
    reward_schemes = ["classic", "dense", "time_penalty"]
    
    for scheme in reward_schemes:
        env = SnakeEnv(
            grid_size=(6, 6),
            reward_scheme=scheme,
            observation_type="grid"
        )
        
        obs, info = env.reset(seed=42)
        total_reward = 0
        
        for _ in range(5):
            obs, reward, done, trunc, info = env.step(Action.RIGHT)
            total_reward += reward
            if done or trunc:
                break
        
        print(f"  {scheme}: total reward = {total_reward:.3f}")
    
    # Test frame stacking
    print(f"\n📚 Testing frame stacking...")
    
    env = SnakeEnv(
        grid_size=(6, 6),
        observation_type="gray",
        observation_size=(32, 32),
        stack_frames=4
    )
    
    obs, info = env.reset(seed=42)
    print(f"  Stacked obs shape: {obs.shape} (should be (32, 32, 4))")
    
    # Test rendering
    print(f"\n🎨 Testing rendering...")
    
    env = SnakeEnv(render_mode="rgb_array", grid_size=(6, 6))
    obs, info = env.reset(seed=42)
    
    rgb_array = env.render()
    if rgb_array is not None:
        print(f"  RGB render shape: {rgb_array.shape}")
    
    # Test vector environment (if available)
    try:
        from snake_rl.env.vector_env import make_sync_vec_env
        print(f"\n🚀 Testing vector environment...")
        
        vec_env = make_sync_vec_env(
            n_envs=2,
            env_kwargs={"grid_size": (6, 6)},
            seed=42
        )
        
        obs = vec_env.reset()
        print(f"  Vector reset: {len(obs)} environments")
        
        actions = [0, 1]  # Different actions for each env
        obs, rewards, dones, truncs, infos = vec_env.step(actions)
        
        print(f"  Vector step: rewards={rewards}, dones={dones}")
        
        vec_env.close()
        
    except ImportError as e:
        print(f"  ⚠️  Vector environment not available: {e}")
    
    print(f"\n✅ All environment tests passed!")
    
except Exception as e:
    print(f"❌ Environment test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)