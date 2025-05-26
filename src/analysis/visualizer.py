"""Visualization tools for OpinionDynamics simulations."""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os
from ..agents.profile import Agent
from ..simulation.engine import SimulationEngine
from ..simulation.analyzer import PopulationMetrics


class SimulationVisualizer:
    """Creates visualizations for simulation results."""
    
    def __init__(self, simulation: Optional[SimulationEngine] = None, 
                 output_dir: str = "visualizations"):
        """Initialize visualizer.
        
        Args:
            simulation: SimulationEngine instance with completed simulation
            output_dir: Directory to save visualizations
        """
        self.simulation = simulation
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def create_all_visualizations(self) -> None:
        """Create all standard visualizations."""
        if not self.simulation:
            raise ValueError("No simulation data available")
        
        print("Creating visualizations...")
        
        # Opinion evolution
        self.plot_opinion_evolution()
        
        # Population state
        self.plot_population_state()
        
        # Interaction network
        self.plot_interaction_network()
        
        # Personality distribution
        self.plot_personality_distribution()
        
        # Influence analysis
        self.plot_influence_analysis()
        
        # Emotional dynamics
        self.plot_emotional_dynamics()
        
        print(f"Visualizations saved to {self.output_dir}/")
    
    def plot_opinion_evolution(self) -> None:
        """Plot how opinions evolved over time."""
        if not self.simulation.metrics_history:
            return
        
        topics = list(list(self.simulation.agents.values())[0].opinions.keys())
        rounds = [m.round_number for m in self.simulation.metrics_history]
        
        # Create figure with subplots
        n_topics = len(topics)
        fig, axes = plt.subplots(n_topics, 2, figsize=(14, 5*n_topics))
        if n_topics == 1:
            axes = axes.reshape(1, -1)
        
        for idx, topic in enumerate(topics):
            # Get trajectories
            trajectories = self.simulation.analyzer.analyze_opinion_trajectories(
                self.simulation.metrics_history, topic)
            
            # Left plot: Mean position with std
            ax1 = axes[idx, 0]
            means = trajectories['mean_positions']
            stds = trajectories['std_positions']
            
            ax1.plot(rounds, means, 'b-', linewidth=2.5, label='Mean Opinion')
            ax1.fill_between(rounds,
                           np.array(means) - np.array(stds),
                           np.array(means) + np.array(stds),
                           alpha=0.3, label='±1 SD')
            
            ax1.set_xlabel('Simulation Round', fontsize=12)
            ax1.set_ylabel('Opinion Position', fontsize=12)
            ax1.set_title(f'{topic.replace("_", " ").title()} - Opinion Evolution', 
                         fontsize=14, fontweight='bold')
            ax1.set_ylim(-100, 100)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Right plot: Polarization and consensus
            ax2 = axes[idx, 1]
            ax2.plot(rounds, trajectories['polarizations'], 'r-', 
                    linewidth=2.5, label='Polarization')
            ax2.plot(rounds, trajectories['consensuses'], 'g-', 
                    linewidth=2.5, label='Consensus')
            
            ax2.set_xlabel('Simulation Round', fontsize=12)
            ax2.set_ylabel('Metric Value', fontsize=12)
            ax2.set_title(f'{topic.replace("_", " ").title()} - Dynamics', 
                         fontsize=14, fontweight='bold')
            ax2.set_ylim(0, 1)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'opinion_evolution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_population_state(self) -> None:
        """Visualize final population state."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Opinion distributions (violin plot)
        ax = axes[0, 0]
        topics = list(list(self.simulation.agents.values())[0].opinions.keys())
        opinion_data = []
        
        for topic in topics:
            positions = [agent.opinions[topic].position 
                        for agent in self.simulation.agents.values()]
            opinion_data.extend([(topic.replace('_', ' ').title(), pos) 
                               for pos in positions])
        
        df = pd.DataFrame(opinion_data, columns=['Topic', 'Position'])
        sns.violinplot(data=df, x='Topic', y='Position', ax=ax)
        ax.set_ylabel('Opinion Position', fontsize=12)
        ax.set_title('Final Opinion Distributions', fontsize=14, fontweight='bold')
        ax.set_ylim(-100, 100)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 2. Personality trait averages
        ax = axes[0, 1]
        traits = ['openness', 'conscientiousness', 'extraversion', 
                 'agreeableness', 'emotionality', 'honesty_humility']
        trait_means = []
        trait_stds = []
        
        for trait in traits:
            values = [getattr(agent.personality, trait) 
                     for agent in self.simulation.agents.values()]
            trait_means.append(np.mean(values))
            trait_stds.append(np.std(values))
        
        x_pos = np.arange(len(traits))
        ax.bar(x_pos, trait_means, yerr=trait_stds, capsize=5, 
               color=sns.color_palette()[1])
        ax.set_xlabel('Personality Traits', fontsize=12)
        ax.set_ylabel('Average Score', fontsize=12)
        ax.set_title('Population Personality Profile', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([t.replace('_', ' ').title() for t in traits], 
                          rotation=45, ha='right')
        ax.set_ylim(0, 100)
        
        # 3. Emotional state heatmap
        ax = axes[1, 0]
        emotional_states = ['valence', 'anxiety', 'confidence', 
                          'social_energy', 'arousal', 'cognitive_load']
        
        # Create matrix of emotional states
        agent_names = [agent.name for agent in self.simulation.agents.values()][:15]
        emotion_matrix = []
        
        for agent in list(self.simulation.agents.values())[:15]:
            agent_emotions = []
            for state in emotional_states:
                value = getattr(agent.emotional_state, state)
                # Normalize to 0-1 scale
                if state == 'valence':
                    normalized = (value + 50) / 100
                else:
                    normalized = value / 100
                agent_emotions.append(normalized)
            emotion_matrix.append(agent_emotions)
        
        sns.heatmap(emotion_matrix, 
                   xticklabels=[s.replace('_', ' ').title() for s in emotional_states],
                   yticklabels=agent_names,
                   cmap='RdYlBu_r', center=0.5, ax=ax)
        ax.set_title('Agent Emotional States', fontsize=14, fontweight='bold')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 4. Opinion certainty vs position scatter
        ax = axes[1, 1]
        for topic in topics[:3]:  # Limit to 3 topics for clarity
            positions = []
            certainties = []
            for agent in self.simulation.agents.values():
                positions.append(agent.opinions[topic].position)
                certainties.append(agent.opinions[topic].certainty)
            
            ax.scatter(positions, certainties, alpha=0.6, s=50,
                      label=topic.replace('_', ' ').title())
        
        ax.set_xlabel('Opinion Position', fontsize=12)
        ax.set_ylabel('Certainty', fontsize=12)
        ax.set_title('Opinion Position vs Certainty', fontsize=14, fontweight='bold')
        ax.set_xlim(-100, 100)
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'population_state.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_interaction_network(self) -> None:
        """Visualize interaction patterns."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. Interaction frequency by agent
        interaction_counts = {}
        for conv in self.simulation.conversations:
            for participant_id in conv.participants:
                interaction_counts[participant_id] = interaction_counts.get(participant_id, 0) + 1
        
        # Sort by interaction count
        sorted_agents = sorted(interaction_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        
        agent_names = [self.simulation.agents[aid].name for aid, _ in sorted_agents]
        counts = [count for _, count in sorted_agents]
        
        y_pos = np.arange(len(agent_names))
        ax1.barh(y_pos, counts, color=sns.color_palette()[2])
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(agent_names, fontsize=10)
        ax1.set_xlabel('Number of Conversations', fontsize=12)
        ax1.set_title('Top 20 Most Active Agents', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # 2. Conversation quality distribution
        qualities = [conv.analysis.overall_quality 
                    for conv in self.simulation.conversations]
        
        ax2.hist(qualities, bins=20, color=sns.color_palette()[3], 
                edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Conversation Quality', fontsize=12)
        ax2.set_ylabel('Number of Conversations', fontsize=12)
        ax2.set_title('Distribution of Conversation Quality', 
                     fontsize=14, fontweight='bold')
        ax2.axvline(np.mean(qualities), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(qualities):.3f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'interaction_network.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_personality_distribution(self) -> None:
        """Create detailed personality distribution plots."""
        traits = ['honesty_humility', 'emotionality', 'extraversion', 
                 'agreeableness', 'conscientiousness', 'openness']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, trait in enumerate(traits):
            ax = axes[idx]
            values = [getattr(agent.personality, trait) 
                     for agent in self.simulation.agents.values()]
            
            # Histogram with KDE
            ax.hist(values, bins=15, density=True, alpha=0.7, 
                   color=sns.color_palette()[idx], edgecolor='black')
            
            # Add KDE
            from scipy import stats
            kde = stats.gaussian_kde(values)
            x_range = np.linspace(0, 100, 100)
            ax.plot(x_range, kde(x_range), 'r-', linewidth=2)
            
            # Add statistics
            mean_val = np.mean(values)
            ax.axvline(mean_val, color='black', linestyle='--', 
                      linewidth=2, alpha=0.7)
            
            ax.set_xlabel('Trait Score', fontsize=11)
            ax.set_ylabel('Density', fontsize=11)
            ax.set_title(f'{trait.replace("_", " ").title()}\n(μ={mean_val:.1f}, σ={np.std(values):.1f})', 
                        fontsize=12, fontweight='bold')
            ax.set_xlim(0, 100)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'personality_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_influence_analysis(self) -> None:
        """Visualize influence patterns."""
        # Get top influencers
        influencers = self.simulation.analyzer.identify_influencers(
            self.simulation.conversations, self.simulation.agents)[:10]
        
        if not influencers:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. Top influencers bar chart
        names = [name for name, _ in influencers]
        scores = [score for _, score in influencers]
        
        y_pos = np.arange(len(names))
        ax1.barh(y_pos, scores, color=sns.color_palette()[4])
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(names, fontsize=11)
        ax1.set_xlabel('Influence Score', fontsize=12)
        ax1.set_title('Top 10 Most Influential Agents', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # 2. Influence vs personality traits
        # Get personality traits of top influencers
        influencer_ids = []
        for conv in self.simulation.conversations:
            for agent_id, changes in conv.state_changes.items():
                total_change = sum([
                    abs(change.get('position', 0))
                    for change in changes.opinion_changes.values()
                ])
                if total_change > 0:
                    partner_id = [pid for pid in conv.participants if pid != agent_id][0]
                    influencer_ids.append(partner_id)
        
        # Analyze traits of influential agents
        if influencer_ids:
            influential_agents = [self.simulation.agents[aid] for aid in set(influencer_ids) 
                                if aid in self.simulation.agents]
            
            trait_correlations = {}
            for trait in ['extraversion', 'openness', 'agreeableness']:
                values = [getattr(agent.personality, trait) for agent in influential_agents]
                trait_correlations[trait] = np.mean(values)
            
            # Compare to population average
            pop_averages = {}
            for trait in trait_correlations:
                pop_values = [getattr(agent.personality, trait) 
                            for agent in self.simulation.agents.values()]
                pop_averages[trait] = np.mean(pop_values)
            
            traits = list(trait_correlations.keys())
            x_pos = np.arange(len(traits))
            width = 0.35
            
            influential_means = [trait_correlations[t] for t in traits]
            population_means = [pop_averages[t] for t in traits]
            
            ax2.bar(x_pos - width/2, influential_means, width, 
                   label='Influential Agents', color=sns.color_palette()[0])
            ax2.bar(x_pos + width/2, population_means, width, 
                   label='Population Average', color=sns.color_palette()[1])
            
            ax2.set_xlabel('Personality Traits', fontsize=12)
            ax2.set_ylabel('Average Score', fontsize=12)
            ax2.set_title('Personality Traits of Influential Agents', 
                         fontsize=14, fontweight='bold')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels([t.replace('_', ' ').title() for t in traits])
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'influence_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_emotional_dynamics(self) -> None:
        """Plot emotional state changes over time."""
        if len(self.simulation.conversations) < 2:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Track emotional states before and after conversations
        emotional_changes = {
            'valence': [],
            'anxiety': [],
            'confidence': [],
            'social_energy': []
        }
        
        for conv in self.simulation.conversations[-50:]:  # Last 50 conversations
            for agent_id, changes in conv.state_changes.items():
                if agent_id in self.simulation.agents:
                    for emotion, delta in changes.emotion_changes.items():
                        if emotion in emotional_changes and delta != 0:
                            emotional_changes[emotion].append(delta)
        
        # 1. Distribution of emotional changes
        ax = axes[0, 0]
        if any(emotional_changes.values()):
            data_to_plot = [changes for emotion, changes in emotional_changes.items() 
                          if changes]
            labels = [emotion.replace('_', ' ').title() 
                     for emotion, changes in emotional_changes.items() if changes]
            
            if data_to_plot:
                ax.boxplot(data_to_plot, labels=labels)
                ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
                ax.set_ylabel('Change in Emotional State', fontsize=12)
                ax.set_title('Distribution of Emotional Changes from Conversations', 
                           fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 2. Average emotional state by personality type
        ax = axes[0, 1]
        
        # Categorize agents by dominant personality trait
        personality_groups = {
            'High Openness': [],
            'High Conscientiousness': [],
            'High Extraversion': [],
            'High Agreeableness': []
        }
        
        for agent in self.simulation.agents.values():
            max_trait = None
            max_value = 0
            
            trait_map = {
                'openness': 'High Openness',
                'conscientiousness': 'High Conscientiousness',
                'extraversion': 'High Extraversion',
                'agreeableness': 'High Agreeableness'
            }
            
            for trait, label in trait_map.items():
                value = getattr(agent.personality, trait)
                if value > max_value and value > 70:
                    max_value = value
                    max_trait = label
            
            if max_trait:
                personality_groups[max_trait].append(agent.emotional_state.valence)
        
        # Plot average valence by personality group
        groups = []
        valences = []
        for group, agent_valences in personality_groups.items():
            if agent_valences:
                groups.append(group)
                valences.append(np.mean(agent_valences))
        
        if groups:
            x_pos = np.arange(len(groups))
            ax.bar(x_pos, valences, color=sns.color_palette()[1])
            ax.set_xticks(x_pos)
            ax.set_xticklabels(groups, rotation=45, ha='right')
            ax.set_ylabel('Average Valence', fontsize=12)
            ax.set_title('Emotional State by Personality Type', 
                        fontsize=14, fontweight='bold')
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Emotional state evolution timeline
        ax = axes[1, 0]
        
        # Sample a few agents to track
        sample_agents = list(self.simulation.agents.values())[:5]
        
        for agent in sample_agents:
            # Create synthetic timeline based on conversations
            valence_timeline = [agent.emotional_baseline.dispositional_affect]
            
            for conv in self.simulation.conversations:
                if agent.id in conv.participants and agent.id in conv.state_changes:
                    changes = conv.state_changes[agent.id]
                    if 'valence' in changes.emotion_changes:
                        new_valence = valence_timeline[-1] + changes.emotion_changes['valence']
                        new_valence = max(-50, min(50, new_valence))
                        valence_timeline.append(new_valence)
                    else:
                        valence_timeline.append(valence_timeline[-1])
            
            if len(valence_timeline) > 1:
                ax.plot(range(len(valence_timeline)), valence_timeline, 
                       alpha=0.7, linewidth=2, label=agent.name[:15])
        
        ax.set_xlabel('Interaction Number', fontsize=12)
        ax.set_ylabel('Valence', fontsize=12)
        ax.set_title('Emotional Valence Evolution (Sample Agents)', 
                    fontsize=14, fontweight='bold')
        ax.set_ylim(-50, 50)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # 4. Correlation between emotions and opinions
        ax = axes[1, 1]
        
        # Get average opinion extremity vs emotional state
        extremities = []
        anxieties = []
        
        for agent in self.simulation.agents.values():
            # Calculate opinion extremity
            extremity = np.mean([abs(op.position) for op in agent.opinions.values()])
            extremities.append(extremity)
            anxieties.append(agent.emotional_state.anxiety)
        
        ax.scatter(extremities, anxieties, alpha=0.6, s=50, color=sns.color_palette()[2])
        
        # Add trend line
        z = np.polyfit(extremities, anxieties, 1)
        p = np.poly1d(z)
        ax.plot(sorted(extremities), p(sorted(extremities)), 
               "r--", alpha=0.8, linewidth=2)
        
        ax.set_xlabel('Average Opinion Extremity', fontsize=12)
        ax.set_ylabel('Anxiety Level', fontsize=12)
        ax.set_title('Opinion Extremity vs Anxiety', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'emotional_dynamics.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_summary_dashboard(self) -> None:
        """Create a single-page summary dashboard."""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Key metrics (text)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axis('off')
        
        final_metrics = self.simulation.metrics_history[-1]
        metrics_text = f"""SIMULATION SUMMARY
        
Agents: {len(self.simulation.agents)}
Rounds: {len(self.simulation.metrics_history)}
Conversations: {len(self.simulation.conversations)}

Final Metrics:
• Polarization: {final_metrics.overall_polarization:.3f}
• Consensus: {final_metrics.overall_consensus:.3f}
• Avg Certainty: {final_metrics.avg_certainty:.1f}
• Avg Mood: {final_metrics.avg_emotional_valence:+.1f}"""
        
        ax1.text(0.1, 0.9, metrics_text, transform=ax1.transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 2. Opinion evolution mini-plot
        ax2 = fig.add_subplot(gs[0, 1:])
        topics = list(list(self.simulation.agents.values())[0].opinions.keys())
        rounds = [m.round_number for m in self.simulation.metrics_history]
        
        for topic in topics[:3]:  # Limit to 3 topics
            trajectories = self.simulation.analyzer.analyze_opinion_trajectories(
                self.simulation.metrics_history, topic)
            ax2.plot(rounds, trajectories['mean_positions'], 
                    linewidth=2, label=topic.replace('_', ' ').title())
        
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Mean Opinion')
        ax2.set_title('Opinion Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Final opinion distribution
        ax3 = fig.add_subplot(gs[1, 0])
        topic = topics[0]  # First topic
        positions = [agent.opinions[topic].position 
                    for agent in self.simulation.agents.values()]
        ax3.hist(positions, bins=20, color=sns.color_palette()[0], 
                edgecolor='black', alpha=0.7)
        ax3.set_xlabel('Opinion Position')
        ax3.set_ylabel('Count')
        ax3.set_title(f'{topic.replace("_", " ").title()} Distribution')
        ax3.set_xlim(-100, 100)
        
        # 4. Personality overview
        ax4 = fig.add_subplot(gs[1, 1])
        traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness']
        trait_means = [np.mean([getattr(agent.personality, trait) 
                               for agent in self.simulation.agents.values()]) 
                      for trait in traits]
        
        ax4.bar(range(len(traits)), trait_means, color=sns.color_palette()[1])
        ax4.set_xticks(range(len(traits)))
        ax4.set_xticklabels([t[:4].upper() for t in traits])
        ax4.set_ylabel('Average Score')
        ax4.set_title('Population Personality')
        ax4.set_ylim(0, 100)
        
        # 5. Top influencers
        ax5 = fig.add_subplot(gs[1, 2])
        influencers = self.simulation.analyzer.identify_influencers(
            self.simulation.conversations, self.simulation.agents)[:5]
        
        if influencers:
            names = [name[:12] for name, _ in influencers]
            scores = [score for _, score in influencers]
            
            y_pos = np.arange(len(names))
            ax5.barh(y_pos, scores, color=sns.color_palette()[2])
            ax5.set_yticks(y_pos)
            ax5.set_yticklabels(names)
            ax5.set_xlabel('Influence')
            ax5.set_title('Top Influencers')
        
        # 6. Polarization trend
        ax6 = fig.add_subplot(gs[2, :2])
        polarizations = [m.overall_polarization for m in self.simulation.metrics_history]
        consensuses = [m.overall_consensus for m in self.simulation.metrics_history]
        
        ax6.plot(rounds, polarizations, 'r-', linewidth=2, label='Polarization')
        ax6.plot(rounds, consensuses, 'g-', linewidth=2, label='Consensus')
        ax6.set_xlabel('Round')
        ax6.set_ylabel('Value')
        ax6.set_title('Polarization and Consensus Over Time')
        ax6.legend()
        ax6.set_ylim(0, 1)
        ax6.grid(True, alpha=0.3)
        
        # 7. Emotional state summary
        ax7 = fig.add_subplot(gs[2, 2])
        emotional_states = ['valence', 'anxiety', 'confidence']
        state_means = []
        
        for state in emotional_states:
            values = [getattr(agent.emotional_state, state) 
                     for agent in self.simulation.agents.values()]
            if state == 'valence':
                state_means.append(np.mean(values))
            else:
                state_means.append(np.mean(values))
        
        x_pos = np.arange(len(emotional_states))
        colors = ['green' if v >= 0 else 'red' for v in state_means]
        ax7.bar(x_pos, state_means, color=colors, alpha=0.7)
        ax7.set_xticks(x_pos)
        ax7.set_xticklabels([s.title() for s in emotional_states])
        ax7.set_ylabel('Average Value')
        ax7.set_title('Population Emotional State')
        ax7.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.suptitle('OpinionDynamics Simulation Dashboard', 
                    fontsize=16, fontweight='bold')
        
        plt.savefig(os.path.join(self.output_dir, 'summary_dashboard.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created summary dashboard: {self.output_dir}/summary_dashboard.png")