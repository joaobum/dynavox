#!/usr/bin/env python3
"""
Test script to demonstrate async parallel conversations in DynaVox.

This script creates a small simulation and clearly shows how conversations
execute in parallel vs sequentially.
"""
import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import QuickSimulation


def test_async_execution():
    """Test and demonstrate async vs sync execution."""
    
    print("=== DynaVox Async Execution Test ===\n")
    
    # Common parameters
    num_agents = 8
    num_rounds = 2
    topics = ['climate_change', 'ai_regulation']
    
    # Test 1: Synchronous execution
    print("1️⃣ SYNCHRONOUS EXECUTION (Sequential)")
    print("-" * 50)
    
    start_time = time.time()
    sim_sync = QuickSimulation(use_mock=True, use_async=False)
    sim_sync.run(
        num_agents=num_agents,
        num_rounds=num_rounds,
        topics=topics,
        interaction_probability=0.5,  # High probability to ensure conversations
        homophily_bias=0.1,  # Low bias to encourage all interactions
        simulation_name='test_sync'
    )
    sync_time = time.time() - start_time
    
    print(f"\n⏱️ Synchronous execution time: {sync_time:.2f} seconds")
    
    # Test 2: Asynchronous execution
    print("\n\n2️⃣ ASYNCHRONOUS EXECUTION (Parallel)")
    print("-" * 50)
    
    start_time = time.time()
    sim_async = QuickSimulation(use_mock=True, use_async=True)
    sim_async.run(
        num_agents=num_agents,
        num_rounds=num_rounds,
        topics=topics,
        interaction_probability=0.5,
        homophily_bias=0.1,
        simulation_name='test_async'
    )
    async_time = time.time() - start_time
    
    print(f"\n⏱️ Asynchronous execution time: {async_time:.2f} seconds")
    
    # Compare results
    print("\n\n📊 COMPARISON")
    print("-" * 50)
    print(f"Synchronous time:  {sync_time:.2f}s")
    print(f"Asynchronous time: {async_time:.2f}s")
    if sync_time > async_time:
        speedup = sync_time / async_time
        print(f"🚀 Speedup: {speedup:.2f}x faster with async!")
    else:
        print("⚠️ No speedup observed (mock LLM may be too fast to show difference)")
    
    # Check conversation counts
    sync_convs = len(sim_sync.sim.conversations)
    async_convs = len(sim_async.sim.conversations)
    print(f"\nConversations completed:")
    print(f"  Synchronous:  {sync_convs}")
    print(f"  Asynchronous: {async_convs}")
    
    print("\n💡 Note: With real LLM API calls, async speedup would be more significant!")
    print("   Mock LLM has minimal delay, so parallelism benefit is reduced.")


if __name__ == "__main__":
    test_async_execution()