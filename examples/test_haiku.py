#!/usr/bin/env python3
"""
Simple test script for Claude 3.5 Haiku model
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import QuickSimulation

def test_haiku():
    """Test Claude 3.5 Haiku with minimal configuration."""
    print("=== Testing Claude 3.5 Haiku ===\n")
    
    # Minimal configuration for testing
    sim = QuickSimulation(model='claude-3-5-haiku', use_async=True)
    
    print("Running minimal test simulation...")
    print("- 3 agents")
    print("- 1 round")
    print("- 2 topics")
    print()
    
    results = sim.run(
        num_agents=3,
        num_rounds=1,
        topics=['climate_change', 'ai_regulation'],
        interaction_probability=0.5,  # High probability to ensure interactions
        simulation_name='test_haiku'
    )
    
    print("\n✅ Test complete!")
    print(f"Results saved to: {results['output_dir']}")
    
    return results

if __name__ == "__main__":
    test_haiku()