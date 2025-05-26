#!/usr/bin/env python3
"""
Example simulation demonstrating the DynaVox framework.

This script shows how to:
1. Initialize agents with diverse personalities
2. Run a multi-round simulation
3. Analyze the results
4. Visualize opinion evolution
"""

import json
import os
import logging
import logging.config
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np

# Import framework components
from src.llm.client import OpenAIClient, MockLLMClient
from src.simulation.engine import SimulationEngine
from src.config import STANDARD_TOPICS, LOGGING_CONFIG, DEFAULT_LOG_LEVEL

# Initialize logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger('opiniondynamics')


def run_opinion_dynamics_simulation(use_mock_llm: bool = False, openai_model: str = None, use_async: bool = False):
    """Run a complete opinion dynamics simulation.
    
    Args:
        use_mock_llm: If True, use mock LLM for testing without API calls
        openai_model: Specific OpenAI model to use (e.g., 'gpt-4', 'gpt-3.5-turbo')
        use_async: If True, use async execution for parallel conversations
    """
    logger.info("Starting DynaVox simulation")
    logger.debug(f"Parameters: use_mock_llm={use_mock_llm}, openai_model={openai_model}, use_async={use_async}")
    print("=== DynaVox Simulation Example ===\n")
    
    # Step 1: Initialize LLM client
    if use_mock_llm:
        logger.info("Initializing mock LLM client")
        print("Using mock LLM client (no API calls)")
        llm_client = MockLLMClient()
    else:
        # Make sure to set your API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY environment variable not set")
            raise ValueError("Please set OPENAI_API_KEY environment variable")
        
        # Use specified model or default
        from src.config import DEFAULT_OPENAI_MODEL, OPENAI_MODELS
        model = openai_model or DEFAULT_OPENAI_MODEL
        
        logger.info(f"Initializing OpenAI client with model: {model}")
        print(f"Using OpenAI model: {model}")
        if model in OPENAI_MODELS:
            print(f"Description: {OPENAI_MODELS[model]}")
        
        llm_client = OpenAIClient(api_key=api_key, model=model)
    
    # Step 2: Create simulation engine
    logger.info("Creating simulation engine")
    sim = SimulationEngine(llm_client, seed=42, use_async=use_async)  # Set seed for reproducibility
    
    # Step 3: Define simulation parameters
    topics = ['climate_change', 'wealth_inequality', 'ai_regulation']
    logger.debug(f"Selected topics: {topics}")
    
    demographics = {
        'age_range': (25, 65),
        'education_distribution': {
            'high_school': 0.3,
            'bachelors': 0.4,
            'masters': 0.2,
            'phd': 0.1
        }
    }
    logger.debug(f"Demographics configuration: {demographics}")
    
    # Step 4: Initialize population
    logger.info(f"Initializing population of 12 agents with topics: {topics}")
    print(f"\nInitializing population with topics: {topics}")
    sim.initialize_population(
        size=12,  # Small population for example
        topics=topics,
        demographics=demographics
    )
    logger.info("Population initialization complete")
    
    # Print initial population summary
    print_population_summary(sim)
    
    # Step 5: Run simulation
    logger.info("Starting simulation with 5 rounds")
    print("\n=== Starting Simulation ===")
    sim.run_simulation(
        rounds=5,
        interaction_probability=0.2,  # 20% chance of interaction
        homophily_bias=0.6,  # Moderate preference for similar others
        max_interactions_per_agent=2
    )
    logger.info("Simulation complete")
    
    # Step 6: Analyze results
    logger.info("Analyzing simulation results")
    print("\n=== Analysis Results ===")
    analyze_simulation_results(sim)
    
    # Step 7: Export results
    logger.info("Exporting results to example_results/")
    sim.export_results("example_results")
    print("\nResults exported to example_results/")
    
    # Step 8: Create visualizations
    if not use_mock_llm:  # Skip visualization for mock runs
        logger.info("Creating visualizations")
        visualize_opinion_evolution(sim)
        visualize_population_state(sim)
    else:
        logger.debug("Skipping visualizations for mock LLM run")
    
    logger.info("Simulation example completed successfully")
    return sim


def print_population_summary(sim: SimulationEngine):
    """Print summary of the agent population."""
    print("\n=== Population Summary ===")
    print(f"Total agents: {len(sim.agents)}")
    
    # Age distribution
    ages = [agent.background.age for agent in sim.agents.values()]
    print(f"Age range: {min(ages)} - {max(ages)} (mean: {np.mean(ages):.1f})")
    
    # Personality distribution
    personality_traits = ['openness', 'conscientiousness', 'extraversion', 
                         'agreeableness', 'emotionality', 'honesty_humility']
    
    print("\nAverage personality traits:")
    for trait in personality_traits:
        values = [getattr(agent.personality, trait) for agent in sim.agents.values()]
        print(f"  {trait}: {np.mean(values):.1f} (σ={np.std(values):.1f})")
    
    # Initial opinion positions
    print("\nInitial opinion positions:")
    for topic in list(sim.agents.values())[0].opinions.keys():
        positions = [agent.opinions[topic].position for agent in sim.agents.values()]
        print(f"  {topic}: mean={np.mean(positions):.1f}, std={np.std(positions):.1f}")


def analyze_simulation_results(sim: SimulationEngine):
    """Analyze and print key findings from the simulation."""
    # Opinion changes
    print("\n1. Opinion Evolution:")
    initial_metrics = sim.metrics_history[0]
    final_metrics = sim.metrics_history[-1]
    
    for topic in initial_metrics.opinion_metrics.keys():
        initial_mean = initial_metrics.opinion_metrics[topic]['mean_position']
        final_mean = final_metrics.opinion_metrics[topic]['mean_position']
        initial_std = initial_metrics.opinion_metrics[topic]['std_position']
        final_std = final_metrics.opinion_metrics[topic]['std_position']
        
        print(f"\n  {topic}:")
        print(f"    Mean position: {initial_mean:.1f} → {final_mean:.1f} "
              f"(Δ={final_mean - initial_mean:+.1f})")
        print(f"    Std deviation: {initial_std:.1f} → {final_std:.1f} "
              f"(Δ={final_std - initial_std:+.1f})")
        print(f"    Polarization: {final_metrics.opinion_metrics[topic]['polarization']:.3f}")
    
    # Overall metrics
    print("\n2. Population Metrics:")
    print(f"  Final polarization: {final_metrics.overall_polarization:.3f}")
    print(f"  Final consensus: {final_metrics.overall_consensus:.3f}")
    print(f"  Average certainty: {final_metrics.avg_certainty:.1f}")
    print(f"  Average mood: {final_metrics.avg_emotional_valence:+.1f}")
    
    # Interaction summary
    print(f"\n3. Interaction Summary:")
    print(f"  Total conversations: {len(sim.conversations)}")
    avg_quality = np.mean([c.analysis.overall_quality for c in sim.conversations])
    print(f"  Average conversation quality: {avg_quality:.3f}")
    avg_duration = np.mean([c.duration_turns for c in sim.conversations])
    print(f"  Average conversation length: {avg_duration:.1f} turns")
    
    # Influential agents
    influencers = sim.analyzer.identify_influencers(sim.conversations, sim.agents)
    if influencers:
        print("\n4. Most Influential Agents:")
        for i, (name, score) in enumerate(influencers[:5], 1):
            print(f"  {i}. {name} (influence score: {score:.1f})")
    
    # Echo chambers
    echo_chambers = sim.analyzer.find_echo_chambers(sim.agents, sim.conversations)
    if echo_chambers:
        print(f"\n5. Echo Chambers Detected: {len(echo_chambers)}")
        for i, chamber in enumerate(echo_chambers, 1):
            print(f"  Chamber {i}: {', '.join(chamber)}")


def visualize_opinion_evolution(sim: SimulationEngine):
    """Create visualizations of opinion evolution."""
    topics = list(sim.agents.values())[0].opinions.keys()
    rounds = [m.round_number for m in sim.metrics_history]
    
    # Create figure with subplots for each topic
    fig, axes = plt.subplots(len(topics), 2, figsize=(12, 4*len(topics)))
    if len(topics) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, topic in enumerate(topics):
        # Get trajectories
        trajectories = sim.analyzer.analyze_opinion_trajectories(
            sim.metrics_history, topic)
        
        # Plot mean position
        ax1 = axes[idx, 0]
        ax1.plot(rounds, trajectories['mean_positions'], 'b-', linewidth=2)
        ax1.fill_between(rounds,
                        np.array(trajectories['mean_positions']) - np.array(trajectories['std_positions']),
                        np.array(trajectories['mean_positions']) + np.array(trajectories['std_positions']),
                        alpha=0.3)
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Mean Position')
        ax1.set_title(f'{topic.replace("_", " ").title()} - Opinion Evolution')
        ax1.set_ylim(-100, 100)
        ax1.grid(True, alpha=0.3)
        
        # Plot polarization and consensus
        ax2 = axes[idx, 1]
        ax2.plot(rounds, trajectories['polarizations'], 'r-', label='Polarization', linewidth=2)
        ax2.plot(rounds, trajectories['consensuses'], 'g-', label='Consensus', linewidth=2)
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Metric Value')
        ax2.set_title(f'{topic.replace("_", " ").title()} - Dynamics')
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('example_results/opinion_evolution.png', dpi=150)
    print("\nSaved opinion evolution plot to example_results/opinion_evolution.png")


def visualize_population_state(sim: SimulationEngine):
    """Visualize final population state."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Personality distribution
    ax = axes[0, 0]
    traits = ['openness', 'conscientiousness', 'extraversion', 
              'agreeableness', 'emotionality', 'honesty_humility']
    trait_values = {trait: [getattr(agent.personality, trait) 
                           for agent in sim.agents.values()] 
                   for trait in traits}
    
    positions = np.arange(len(traits))
    width = 0.35
    
    means = [np.mean(trait_values[trait]) for trait in traits]
    stds = [np.std(trait_values[trait]) for trait in traits]
    
    ax.bar(positions, means, width, yerr=stds, capsize=5)
    ax.set_xlabel('Personality Traits')
    ax.set_ylabel('Average Score')
    ax.set_title('Population Personality Profile')
    ax.set_xticks(positions)
    ax.set_xticklabels([t.replace('_', ' ').title() for t in traits], rotation=45)
    ax.set_ylim(0, 100)
    
    # 2. Opinion distribution (final)
    ax = axes[0, 1]
    topics = list(sim.agents.values())[0].opinions.keys()
    topic_data = []
    
    for topic in topics:
        positions = [agent.opinions[topic].position for agent in sim.agents.values()]
        topic_data.append(positions)
    
    ax.boxplot(topic_data, labels=[t.replace('_', ' ').title() for t in topics])
    ax.set_ylabel('Opinion Position')
    ax.set_title('Final Opinion Distributions')
    ax.set_ylim(-100, 100)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # 3. Emotional state distribution
    ax = axes[1, 0]
    emotional_states = ['valence', 'anxiety', 'confidence', 'social_energy']
    state_values = {state: [getattr(agent.emotional_state, state)
                           for agent in sim.agents.values()]
                   for state in emotional_states}
    
    state_data = [state_values[state] for state in emotional_states]
    ax.boxplot(state_data, labels=[s.replace('_', ' ').title() for s in emotional_states])
    ax.set_ylabel('Score')
    ax.set_title('Final Emotional States')
    
    # 4. Interaction network summary
    ax = axes[1, 1]
    interaction_counts = {}
    for conv in sim.conversations:
        for participant in conv.participants:
            interaction_counts[participant] = interaction_counts.get(participant, 0) + 1
    
    agent_names = [sim.agents[aid].name for aid in interaction_counts.keys()]
    counts = list(interaction_counts.values())
    
    y_pos = np.arange(len(agent_names))
    ax.barh(y_pos, counts)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(agent_names, fontsize=8)
    ax.set_xlabel('Number of Conversations')
    ax.set_title('Agent Interaction Frequency')
    
    plt.tight_layout()
    plt.savefig('example_results/population_state.png', dpi=150)
    print("Saved population state visualization to example_results/population_state.png")


def load_and_analyze_results(results_dir: str = "example_results"):
    """Load and analyze previously saved results."""
    print(f"\n=== Loading Results from {results_dir} ===")
    
    # Load agent data
    with open(os.path.join(results_dir, 'agents.json'), 'r') as f:
        agents_data = json.load(f)
    
    print(f"Loaded {len(agents_data)} agents")
    
    # Load metrics history
    with open(os.path.join(results_dir, 'metrics_history.json'), 'r') as f:
        metrics_data = json.load(f)
    
    print(f"Loaded {len(metrics_data)} rounds of metrics")
    
    # Load analysis
    with open(os.path.join(results_dir, 'analysis.json'), 'r') as f:
        analysis_data = json.load(f)
    
    print("\nKey Findings:")
    print(f"- Final polarization: {analysis_data['final_metrics']['polarization']:.3f}")
    print(f"- Final consensus: {analysis_data['final_metrics']['consensus']:.3f}")
    print(f"- Top influencer: {analysis_data['top_influencers'][0][0]} "
          f"(score: {analysis_data['top_influencers'][0][1]:.1f})")
    print(f"- Echo chambers found: {len(analysis_data['echo_chambers'])}")
    
    return agents_data, metrics_data, analysis_data


if __name__ == "__main__":
    import argparse
    from src.config import OPENAI_MODELS
    
    parser = argparse.ArgumentParser(description='Run DynaVox simulation example')
    parser.add_argument('--mock', action='store_true', 
                       help='Use mock LLM (no API calls)')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze existing results')
    parser.add_argument('--model', type=str, default=None,
                       help=f'OpenAI model to use. Options: {", ".join(OPENAI_MODELS.keys())}')
    parser.add_argument('--list-models', action='store_true',
                       help='List available OpenAI models and exit')
    parser.add_argument('--async', action='store_true',
                       help='Use async execution for parallel conversations')
    parser.add_argument('--log-level', type=str, default=DEFAULT_LOG_LEVEL,
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help=f'Set logging level (default: {DEFAULT_LOG_LEVEL})')
    
    args = parser.parse_args()
    
    # Set log level if specified
    if args.log_level:
        numeric_level = getattr(logging, args.log_level)
        # Set log level for all loggers
        logging.getLogger('opiniondynamics').setLevel(numeric_level)
        logging.getLogger('opiniondynamics.simulation').setLevel(numeric_level)
        logging.getLogger('opiniondynamics.llm').setLevel(numeric_level)
        logging.getLogger('opiniondynamics.interactions').setLevel(numeric_level)
        logging.getLogger('opiniondynamics.agents').setLevel(numeric_level)
        logger.info(f"Log level set to {args.log_level}")
    
    if args.list_models:
        print("\nAvailable OpenAI Models:")
        print("-" * 60)
        for model, description in OPENAI_MODELS.items():
            print(f"{model:<25} {description}")
        print("\nUsage: python example_simulation.py --model <model-name>")
        exit(0)
    
    if args.analyze_only:
        # Just load and analyze existing results
        load_and_analyze_results()
    else:
        # Run new simulation
        sim = run_opinion_dynamics_simulation(use_mock_llm=args.mock, openai_model=args.model, use_async=getattr(args, 'async', False))
        
        print("\n=== Simulation Complete ===")
        print("You can now:")
        print("1. Examine the results in example_results/")
        print("2. Run with --analyze-only to reload results")
        print("3. Modify parameters and run again")
        print("4. Use the simulation object for further analysis")