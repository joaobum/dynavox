#!/usr/bin/env python3
"""
Test that logging levels work correctly - DEBUG messages should not appear at INFO level
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import QuickSimulation

def test_info_level():
    """Test INFO level logging - should not show LLM prompts."""
    print("=== Testing INFO Level Logging ===")
    print("LLM prompts and responses should NOT appear in console\n")
    
    sim = QuickSimulation(use_mock=True, log_level='INFO')
    
    results = sim.run(
        num_agents=2,
        num_rounds=1,
        topics=['climate_change'],
        interaction_probability=1.0,  # Ensure interaction
        simulation_name='test_info_level'
    )
    
    print("\n✅ Test complete!")
    print(f"Check log file for DEBUG details: {results['output_dir']}/logs/")
    
def test_debug_level():
    """Test DEBUG level logging - should show everything."""
    print("\n\n=== Testing DEBUG Level Logging ===")
    print("LLM prompts and responses SHOULD appear in console\n")
    
    sim = QuickSimulation(use_mock=True, log_level='DEBUG')
    
    results = sim.run(
        num_agents=2,
        num_rounds=1,
        topics=['climate_change'],
        interaction_probability=1.0,  # Ensure interaction
        simulation_name='test_debug_level'
    )
    
    print("\n✅ Test complete!")

if __name__ == "__main__":
    test_info_level()
    # Uncomment to also test DEBUG level
    # test_debug_level()