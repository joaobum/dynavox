#!/usr/bin/env python3
"""Final verification of logging behavior"""
import sys
import os
import logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test logging directly
print("=== Direct Logger Test ===")
from src.utils.enhanced_logging import setup_enhanced_logging

# Set up INFO level logging
output_dir, log_path = setup_enhanced_logging(
    output_dir="test_logs",
    model_name="test",
    num_agents=2,
    num_rounds=1,
    interaction_prob=0.5,
    homophily=0.5,
    is_mock=True,
    log_level='INFO'
)

# Get loggers
llm_logger = logging.getLogger('dynavox.llm')
sim_logger = logging.getLogger('dynavox.simulation')

# Test messages
print("\nTesting at INFO level:")
print(f"LLM logger level: {logging.getLevelName(llm_logger.level)}")
print(f"Sim logger level: {logging.getLevelName(sim_logger.level)}")

llm_logger.info("This INFO message should appear")
llm_logger.debug("This DEBUG message should NOT appear in console")
sim_logger.debug("This sim DEBUG should NOT appear in console")

# Check log file
print(f"\nChecking log file: {log_path}")
with open(log_path, 'r') as f:
    content = f.read()
    debug_count = content.count("[DEBUG")
    info_count = content.count("[INFO")
    print(f"DEBUG entries in file: {debug_count}")
    print(f"INFO entries in file: {info_count}")
    
# Clean up
import shutil
shutil.rmtree("test_logs")