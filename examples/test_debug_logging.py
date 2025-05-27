#!/usr/bin/env python3
"""Test DEBUG logging to ensure prompts are captured"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import QuickSimulation

# Test with DEBUG level
print("=== DEBUG Level Test ===")
sim = QuickSimulation(use_mock=True, log_level='DEBUG')
results = sim.run(num_agents=2, num_rounds=1, topics=['test'], simulation_name='debug_test')
print(f"\nLog file: {results['output_dir']}/logs/")