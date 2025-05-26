#!/usr/bin/env python3
"""Test script to verify DynaVox installation."""

import sys
print(f"Python version: {sys.version}")
print()

# Test core imports
try:
    import numpy as np
    print(f"✓ NumPy {np.__version__} imported successfully")
except ImportError as e:
    print(f"✗ Failed to import NumPy: {e}")

try:
    import pandas as pd
    print(f"✓ Pandas {pd.__version__} imported successfully")
except ImportError as e:
    print(f"✗ Failed to import Pandas: {e}")

try:
    import matplotlib
    print(f"✓ Matplotlib {matplotlib.__version__} imported successfully")
except ImportError as e:
    print(f"✗ Failed to import Matplotlib: {e}")

try:
    import seaborn as sns
    print(f"✓ Seaborn {sns.__version__} imported successfully")
except ImportError as e:
    print(f"✗ Failed to import Seaborn: {e}")

# Test LLM imports
try:
    import openai
    print(f"✓ OpenAI {openai.__version__} imported successfully")
except ImportError as e:
    print(f"✗ Failed to import OpenAI: {e}")

try:
    import anthropic
    print(f"✓ Anthropic {anthropic.__version__} imported successfully")
except ImportError as e:
    print(f"✗ Failed to import Anthropic: {e}")

# Test DynaVox imports
print("\nTesting DynaVox imports:")
try:
    from src.agents import Agent, PersonalityTraits, Background
    print("✓ Agent components imported successfully")
except ImportError as e:
    print(f"✗ Failed to import Agent components: {e}")

try:
    from src.llm import create_llm_client, MockLLMClient
    print("✓ LLM components imported successfully")
except ImportError as e:
    print(f"✗ Failed to import LLM components: {e}")

try:
    from src.simulation import SimulationEngine
    print("✓ Simulation engine imported successfully")
except ImportError as e:
    print(f"✗ Failed to import Simulation engine: {e}")

# Test basic functionality
print("\nTesting basic functionality:")
try:
    from src.llm import MockLLMClient
    from src.simulation.engine import SimulationEngine
    
    # Create mock client
    client = MockLLMClient()
    print("✓ Created mock LLM client")
    
    # Create simulation engine
    sim = SimulationEngine(client)
    print("✓ Created simulation engine")
    
    # Test basic agent creation
    from src.agents.profile import PersonalityTraits, Agent
    traits = PersonalityTraits(50, 50, 50, 50, 50, 50)
    print("✓ Created personality traits")
    
    print("\n✅ All basic tests passed! DynaVox is ready to use.")
    
except Exception as e:
    print(f"\n✗ Error during functionality test: {e}")
    import traceback
    traceback.print_exc()

print("\nTo run the example simulation:")
print("  python example_simulation.py --mock  # Test without API")
print("  python example_simulation.py --list-models  # See available models")
print("  python example_simulation.py --model gpt-4o  # Run with specific model")