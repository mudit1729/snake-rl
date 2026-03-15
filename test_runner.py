#!/usr/bin/env python3
"""
Simple test runner for development without Poetry installed.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

try:
    # Test basic imports
    from snake_rl.sim.engine import SnakeEngine, Action, CellType
    from snake_rl.sim.renderer import SnakeRenderer
    print("✅ All imports successful")
    
    # Test basic functionality
    engine = SnakeEngine(grid_size=(10, 10))
    state = engine.reset(seed=42)
    print(f"✅ Engine initialized: grid shape {state.grid.shape}")
    
    # Take a few steps
    for i in range(5):
        state, reward, done, info = engine.step(Action.RIGHT)
        print(f"Step {i+1}: score={state.score}, done={done}")
        if done:
            break
    
    # Test renderer (without pygame)
    renderer = SnakeRenderer()
    array_obs = renderer.render_array(state)
    print(f"✅ Renderer works: array shape {array_obs.shape}")
    
    # Test observation rendering
    try:
        obs = renderer.render_observation(state, output_size=(84, 84), channels="gray")
        print(f"✅ Observation rendering: shape {obs.shape}")
    except ImportError as e:
        print(f"⚠️  PIL not available for observation resizing: {e}")
    
    print("\n🎉 Basic functionality test passed!")
    print("Ready to proceed with Gymnasium environment implementation.")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)