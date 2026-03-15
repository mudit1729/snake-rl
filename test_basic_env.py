#!/usr/bin/env python3
"""
Basic environment test without gymnasium dependency.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

# Test imports without gymnasium
try:
    from snake_rl.sim.engine import SnakeEngine, Action, CellType
    from snake_rl.sim.renderer import SnakeRenderer
    print("✅ Core simulator imports successful")
    
    # Test engine functionality
    engine = SnakeEngine(grid_size=(8, 8))
    state = engine.reset(seed=42)
    print(f"✅ Engine working: grid {state.grid.shape}, score {state.score}")
    
    # Test renderer
    renderer = SnakeRenderer()
    array_obs = renderer.render_array(state)
    print(f"✅ Renderer working: array shape {array_obs.shape}")
    
    # Test observation rendering
    try:
        obs = renderer.render_observation(state, output_size=(64, 64), channels="gray")
        print(f"✅ Observation rendering: shape {obs.shape}")
    except Exception as e:
        print(f"⚠️  Observation rendering limited: {e}")
    
    # Test multiple reward computations
    print(f"\n💰 Testing reward schemes manually...")
    
    engine = SnakeEngine(grid_size=(6, 6))
    state = engine.reset(seed=42)
    
    # Simulate different reward schemes
    def compute_classic_reward(raw_reward, info):
        return raw_reward
    
    def compute_dense_reward(raw_reward, info, state):
        reward = raw_reward
        if "food_eaten" not in info:
            reward -= 0.01
            # Distance reward
            head_pos = state.snake_positions[0]
            food_pos = state.food_position
            distance = abs(head_pos[0] - food_pos[0]) + abs(head_pos[1] - food_pos[1])
            max_distance = 12  # 6 + 6
            distance_reward = 0.1 * (1.0 - distance / max_distance)
            reward += distance_reward
        return reward
    
    # Take some steps and test reward computation
    for step in range(5):
        action = Action.RIGHT
        state, raw_reward, done, info = engine.step(action)
        
        classic_reward = compute_classic_reward(raw_reward, info)
        dense_reward = compute_dense_reward(raw_reward, info, state)
        
        print(f"  Step {step+1}: raw={raw_reward:.3f}, classic={classic_reward:.3f}, dense={dense_reward:.3f}")
        
        if done:
            break
    
    # Test frame stacking simulation
    print(f"\n📚 Testing frame stacking concept...")
    
    frame_buffer = []
    stack_frames = 3
    
    engine = SnakeEngine(grid_size=(5, 5))
    state = engine.reset(seed=123)
    
    for step in range(5):
        # Get current observation
        obs = renderer.render_array(state)
        
        # Add to buffer
        frame_buffer.append(obs)
        
        # Maintain buffer size
        if len(frame_buffer) > stack_frames:
            frame_buffer.pop(0)
        
        # Pad if necessary
        while len(frame_buffer) < stack_frames:
            frame_buffer.insert(0, obs)
        
        print(f"  Step {step+1}: buffer size {len(frame_buffer)}, obs shape {obs.shape}")
        
        # Take step
        state, _, done, _ = engine.step(Action.RIGHT)
        if done:
            break
    
    # Test observation space concepts
    print(f"\n🔍 Testing observation space concepts...")
    
    # Simulate different observation types
    state = engine.reset(seed=456)
    
    # Grid observation (raw)
    grid_obs = renderer.render_array(state)
    print(f"  Grid obs: shape {grid_obs.shape}, dtype {grid_obs.dtype}")
    
    # RGB observation  
    try:
        rgb_obs = renderer.render_rgb(state, scale=1)
        print(f"  RGB obs: shape {rgb_obs.shape}, dtype {rgb_obs.dtype}")
    except Exception as e:
        print(f"  ⚠️  RGB obs not available: {e}")
    
    # Grayscale observation
    try:
        gray_obs = renderer.render_observation(state, output_size=(32, 32), channels="gray")
        print(f"  Gray obs: shape {gray_obs.shape}, dtype {gray_obs.dtype}")
    except Exception as e:
        print(f"  ⚠️  Gray obs limited: {e}")
    
    print(f"\n🎉 Core environment components working!")
    print(f"Ready for gymnasium wrapper (when gymnasium is available)")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)