#!/usr/bin/env python3
"""Simple test to debug OpenAI integration."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.llm.client import OpenAIClient
from src.simulation.engine import SimulationEngine

def test_simple_simulation():
    """Run a minimal simulation to test OpenAI integration."""
    print("=== Simple DynaVox Test ===\n")
    
    # Step 1: Initialize LLM client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    print("Initializing OpenAI client with gpt-4o-mini...")
    llm_client = OpenAIClient(api_key=api_key, model="gpt-4o-mini")
    
    # Step 2: Create simulation engine with minimal settings
    print("Creating simulation engine...")
    sim = SimulationEngine(llm_client, seed=42)
    
    # Step 3: Initialize small population
    print("\nInitializing population with 2 agents...")
    topics = ['climate_change']
    sim.initialize_population(
        size=2,  # Just 2 agents for testing
        topics=topics
    )
    
    print(f"Created {len(sim.agents)} agents successfully")
    for agent in sim.agents.values():
        print(f"  - {agent.name} ({agent.background.age}, {agent.background.occupation})")
    
    # Step 4: Run one round
    print("\n=== Running Single Round ===")
    sim.run_simulation(
        rounds=1,
        interaction_probability=1.0,  # Force interaction
        homophily_bias=0.0,  # No bias
        max_interactions_per_agent=1
    )
    
    print("\n=== Test Complete ===")
    return sim

if __name__ == "__main__":
    try:
        sim = test_simple_simulation()
        print("Success! Simulation completed without errors.")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()