#!/usr/bin/env python3
"""
Test enhanced logging with unicode and real-time updates.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import QuickSimulation

def test_enhanced_logging():
    """Test the enhanced logging system."""
    print("=== Testing Enhanced Logging ===\n")
    
    # Create simulation with minimal config
    sim = QuickSimulation(use_mock=True, log_level='DEBUG')
    
    print("Running test simulation with enhanced logging...")
    
    results = sim.run(
        num_agents=4,
        num_rounds=1,
        topics=['climate_change', 'ai_regulation'],
        interaction_probability=0.8,  # High to ensure interactions
        homophily_bias=0.2,  # Low bias
        simulation_name='test_logging'
    )
    
    print("\n✅ Test complete!")
    print(f"\n📁 Check the results at: {results['output_dir']}")
    print(f"📝 Log file: {results.get('log_file', 'Check logs folder')}")
    print("\n💡 The log file should contain:")
    print("   - Unicode indicators for different event types")
    print("   - All console output")
    print("   - DEBUG level details")
    print("\n📊 Real-time data files should be in:")
    print(f"   - {results['output_dir']}/data/agents_realtime.json")
    print(f"   - {results['output_dir']}/data/conversations_realtime.json")
    print(f"   - {results['output_dir']}/data/metrics_history_realtime.json")
    
    return results

if __name__ == "__main__":
    test_enhanced_logging()