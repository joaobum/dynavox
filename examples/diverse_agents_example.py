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
    
    # Create simulation with specific model
    # Using mock mode for demonstration - change to False to use real LLM
    use_mock = False
    
    if use_mock:
        print("🔧 Running in MOCK mode (no API calls)")
        sim = QuickSimulation(use_mock=True)
    else:
        # For real simulation, use a cost-effective model
        print("💡 Using GPT-4o-mini for cost-effective simulation")
        sim = QuickSimulation(model='gpt-4o-mini')
    
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
    homophily_bias = 0.2
    
    # Higher interaction probability to ensure diverse encounters
    interaction_probability = 0.25
    
    # More interactions per agent per round
    max_interactions_per_agent = 3
    
    print("🔧 Configuration:")
    print(f"   • Agents: 20 (diverse backgrounds)")
    print(f"   • Rounds: 15")
    print(f"   • Homophily bias: {homophily_bias} (low - encourages cross-group interaction)")
    print(f"   • Interaction probability: {interaction_probability}")
    print(f"   • Max interactions per agent: {max_interactions_per_agent}")
    print()
    
    # For even more control over diversity, you could use SimulationEngine directly:
    if False:  # Set to True to use advanced configuration
        llm_client = create_llm_client('mock:test' if use_mock else 'gpt-4o-mini')
        sim_engine = SimulationEngine(llm_client)
        
        # Force specific personality distributions
        personality_biases = [
            'high openness',           # 4 agents - creative, curious
            'high conscientiousness',  # 4 agents - organized, disciplined
            'high agreeableness',      # 3 agents - cooperative, trusting
            'high extraversion',       # 3 agents - outgoing, energetic
            'low agreeableness',       # 3 agents - competitive, skeptical
            'balanced'                 # 3 agents - moderate traits
        ]
        
        # Create agents with forced diversity
        for i in range(20):
            bias = personality_biases[i % len(personality_biases)]
            sim_engine.initialize_population(
                size=1,
                topics=topics,
                personality_biases=[bias],
                age_range=(18 + (i * 3), 25 + (i * 3))  # Spread ages
            )
    
    # Run the simulation
    print("🚀 Starting simulation...\n")
    
    results = sim.run(
        num_agents=20,
        num_rounds=15,
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
    # Run the diverse agents simulation
    results = run_diverse_simulation()
    
    # Optional: Analyze cross-group interactions
    # analyze_cross_group_interactions(results)
    
    print("\n💡 Tip: Compare these results with a high homophily simulation")
    print("   to see how social mixing affects opinion dynamics!")