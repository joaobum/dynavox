#!/usr/bin/env python3
"""
Test script using OpenAI to verify the system works
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import QuickSimulation

def test_openai():
    """Test with OpenAI GPT-4o-mini."""
    print("=== Testing with OpenAI GPT-4o-mini ===\n")
    
    # Check if OpenAI key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ No OPENAI_API_KEY found in environment")
        print("Please set your OpenAI API key in the .env file")
        return
    
    # Minimal configuration for testing
    sim = QuickSimulation(model='gpt-4o-mini', use_async=True)
    
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
        simulation_name='test_openai'
    )
    
    print("\n✅ Test complete!")
    print(f"Results saved to: {results['output_dir']}")
    
    return results

if __name__ == "__main__":
    test_openai()