#!/usr/bin/env python3
"""
Diverse Agents Example - DynaVox Framework

This example demonstrates running a simulation with highly diverse agents
and low homophily bias to encourage cross-group interactions.

The simulation creates 20 agents with varied:
- Personality traits (using different personality biases)
- Age ranges (from young adults to elderly)
- Socioeconomic backgrounds
- Educational levels
- Occupations across different sectors
"""
import sys
import os
import logging

# Add parent directory to path to import src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import QuickSimulation, SimulationEngine, create_llm_client
from src.config import AGENT_GENERATION_PARAMS

# Suppress HTTP request logging from OpenAI/Anthropic
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("anthropic").setLevel(logging.WARNING)


def run_diverse_simulation():
    """Run simulation with diverse agents and low homophily bias."""
    
    print("=== DynaVox Diverse Agents Simulation ===\n")
    
    # Get global settings from command line args
    use_mock = globals().get('use_mock', True)
    use_async = globals().get('USE_ASYNC', False)
    model_choice = globals().get('MODEL_CHOICE', None)
    
    if use_mock:
        print("🔧 Running in MOCK mode (no API calls)")
        sim = QuickSimulation(use_mock=True, use_async=use_async)
    else:
        # For real simulation, use specified model or default
        if model_choice:
            print(f"💡 Using {model_choice} model")
            sim = QuickSimulation(model=model_choice, use_async=use_async)
        else:
            print("💡 Using GPT-4o-mini for cost-effective simulation")
            sim = QuickSimulation(model='gpt-4o-mini', use_async=use_async)
    
    if use_async:
        print("⚡ Async mode enabled - conversations will run in parallel")
    
    # Define diverse topics that different groups might have varying opinions on
    topics = [
        'wealth_inequality',      # Economic views vary by class
        'climate_change',         # Generational and educational differences
        'ai_regulation',          # Tech exposure varies by profession
        'universal_healthcare',   # Personal impact varies by situation
        'remote_work'            # Professional relevance varies
    ]
    
    print(f"📋 Topics: {', '.join(topics)}\n")
    
    # Configure for diversity
    # Low homophily bias (0.3) means agents are more likely to interact
    # with those who are different from them
    homophily_bias = 0.15
    
    # Higher interaction probability to ensure diverse encounters
    interaction_probability = 0.25
    
    # More interactions per agent per round
    max_interactions_per_agent = 3
    
    # Configuration based on mode
    num_rounds = 3 if use_mock else 15
    num_agents = 20  # Always use 20 agents
    
    print("🔧 Configuration:")
    print(f"   • Agents: {num_agents} (diverse backgrounds)")
    print(f"   • Rounds: {num_rounds}")
    print(f"   • Homophily bias: {homophily_bias} (low - encourages cross-group interaction)")
    print(f"   • Interaction probability: {interaction_probability}")
    print(f"   • Max interactions per agent: {max_interactions_per_agent}")
    print()
    
    # For even more control over diversity, you could use SimulationEngine directly:
    # if False:  # Set to True to use advanced configuration
    #     llm_client = create_llm_client('mock:test' if use_mock else 'gpt-4o-mini')
    #     sim_engine = SimulationEngine(llm_client, use_async=use_async)
    #     
    #     # Force specific personality distributions
    #     personality_biases = [
    #         'high openness',           # 4 agents - creative, curious
    #         'high conscientiousness',  # 4 agents - organized, disciplined
    #         'high agreeableness',      # 3 agents - cooperative, trusting
    #         'high extraversion',       # 3 agents - outgoing, energetic
    #         'low agreeableness',       # 3 agents - competitive, skeptical
    #         'balanced'                 # 3 agents - moderate traits
    #     ]
    #     
    #     # Create agents with forced diversity
    #     for i in range(20):
    #         bias = personality_biases[i % len(personality_biases)]
    #         sim_engine.initialize_population(
    #             size=1,
    #             topics=topics,
    #             personality_biases=[bias],
    #             age_range=(18 + (i * 3), 25 + (i * 3))  # Spread ages
    #         )
    
    # Run the simulation
    print("🚀 Starting simulation...\n")
    
    results = sim.run(
        num_agents=num_agents,
        num_rounds=num_rounds,
        topics=topics,
        interaction_probability=interaction_probability,
        homophily_bias=homophily_bias,
        max_interactions_per_agent=max_interactions_per_agent,
        simulation_name='diverse_agents'
    )
    
    print("\n✅ Simulation complete!")
    print("\n📊 Key insights to look for in the results:")
    print("   • Cross-group opinion convergence (or lack thereof)")
    print("   • Bridge-builders who connect different groups")
    print("   • Persistent echo chambers despite low homophily")
    print("   • Topics where consensus emerges vs. polarization")
    
    print(f"\n📁 Results saved to: {results['output_dir']}")
    print(f"   • 📊 Dashboard: {results['summary_dashboard']}")
    print(f"   • 📄 Full report: {results['full_report']}")
    print(f"   • 📈 Visualizations: {results['visualizations']}")
    
    # Analyze diversity in final population
    if not use_mock:
        print("\n🔍 Population Diversity Analysis:")
        # This would analyze the actual agent backgrounds
        # Mock mode generates less diverse agents
    
    return results


def analyze_cross_group_interactions(results):
    """Analyze interactions between different demographic groups."""
    # Load conversation data
    import json
    
    conversations_path = os.path.join(results['data'], 'conversations.json')
    if os.path.exists(conversations_path):
        with open(conversations_path, 'r') as f:
            conversations = json.load(f)
        
        print(f"\n📊 Cross-Group Interaction Analysis:")
        print(f"   Total conversations: {len(conversations)}")
        
        # You could analyze:
        # - Age differences in conversation pairs
        # - Socioeconomic diversity in interactions
        # - Opinion distance between conversing agents
        # - Successful influence across demographic boundaries


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run DynaVox diverse agents simulation')
    parser.add_argument('--async', action='store_true', 
                       help='Use async execution for parallel conversations')
    parser.add_argument('--real', action='store_true',
                       help='Use real LLM instead of mock')
    parser.add_argument('--model', type=str, default=None,
                       help='LLM model to use (e.g., gpt-4o-mini, claude-3-5-haiku)')
    
    args = parser.parse_args()
    
    # Override the default mock setting if --real is specified
    if args.real:
        use_mock = False
    else:
        use_mock = True
    
    # Store model choice globally
    MODEL_CHOICE = args.model
    
    # Modify the run_diverse_simulation function call
    import inspect
    
    # Temporarily store the async flag
    USE_ASYNC = getattr(args, 'async')
    
    # Run the diverse agents simulation
    results = run_diverse_simulation()
    
    # Optional: Analyze cross-group interactions
    # analyze_cross_group_interactions(results)
    
    print("\n💡 Tip: Compare these results with a high homophily simulation")
    print("   to see how social mixing affects opinion dynamics!")