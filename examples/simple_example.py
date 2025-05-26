#!/usr/bin/env python3
"""
Simplified example demonstrating the DynaVox framework.

This example shows how easy it is to run a simulation with the new
automatic configuration and analysis features.
"""
import sys
import os

# Add parent directory to path to import src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The simplest possible simulation - just 3 lines!
from src import QuickSimulation

# Create simulation (automatically loads API keys from .env)
# Using mock mode for demonstration - remove use_mock=True to use real LLM
sim = QuickSimulation()

# Run with default settings and generate all outputs
results = sim.quick_run('test')  # 'test' is fast (5 agents, 3 rounds)

print("\n✅ Simulation complete! Check out your results:")
print(f"   📊 Dashboard: {results['summary_dashboard']}")
print(f"   📄 Report: {results['full_report']}")


# For more control, you can customize parameters:
"""
# Custom example with specific model and settings
sim = QuickSimulation(model='gpt-4o-mini')  # Use budget-friendly model

results = sim.run(
    num_agents=30,
    num_rounds=15,
    topics=['climate_change', 'ai_regulation'],
    interaction_probability=0.2,
    homophily_bias=0.7
)
"""

# Or use different presets:
"""
sim.quick_run('small')   # 20 agents, 10 rounds
sim.quick_run('medium')  # 50 agents, 20 rounds  
sim.quick_run('large')   # 100 agents, 30 rounds
"""

# You can also use mock mode for testing without API calls:
"""
sim = QuickSimulation(use_mock=True)
results = sim.quick_run('test')
"""