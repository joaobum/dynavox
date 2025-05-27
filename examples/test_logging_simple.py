#!/usr/bin/env python3
"""Simple test to verify logging levels"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import QuickSimulation

# Test with INFO level
print("=== INFO Level Test ===")
sim = QuickSimulation(use_mock=True, log_level='INFO')
results = sim.run(num_agents=2, num_rounds=1, topics=['test'], simulation_name='info_test')
print(f"\nLog file: {results['output_dir']}/logs/")

# Count prompts in log
import glob
log_files = glob.glob(f"{results['output_dir']}/logs/*.log")
if log_files:
    with open(log_files[0], 'r') as f:
        content = f.read()
        prompt_count = content.count("Mock LLM Prompt:")
        print(f"LLM prompts in log file: {prompt_count}")
        console_prompts = content.count("[DEBUG") 
        print(f"DEBUG entries in log: {console_prompts}")