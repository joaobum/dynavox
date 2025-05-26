"""Report generation for OpinionDynamics simulations."""
import os
import json
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from ..simulation.engine import SimulationEngine
from ..agents.profile import Agent


class SimulationReporter:
    """Generates comprehensive reports from simulation results."""
    
    def __init__(self, simulation: Optional[SimulationEngine] = None,
                 output_dir: str = "reports"):
        """Initialize reporter.
        
        Args:
            simulation: SimulationEngine instance with completed simulation
            output_dir: Directory to save reports
        """
        self.simulation = simulation
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_full_report(self) -> str:
        """Generate comprehensive simulation report."""
        if not self.simulation:
            raise ValueError("No simulation data available")
        
        report_path = os.path.join(self.output_dir, "simulation_report.md")
        
        with open(report_path, 'w') as f:
            # Header
            f.write("# OpinionDynamics Simulation Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write(self._generate_executive_summary())
            f.write("\n\n")
            
            # Simulation Parameters
            f.write("## Simulation Parameters\n\n")
            f.write(self._generate_parameters_section())
            f.write("\n\n")
            
            # Population Overview
            f.write("## Population Overview\n\n")
            f.write(self._generate_population_overview())
            f.write("\n\n")
            
            # Opinion Dynamics Results
            f.write("## Opinion Dynamics Results\n\n")
            f.write(self._generate_opinion_results())
            f.write("\n\n")
            
            # Interaction Analysis
            f.write("## Interaction Analysis\n\n")
            f.write(self._generate_interaction_analysis())
            f.write("\n\n")
            
            # Influence Patterns
            f.write("## Influence Patterns\n\n")
            f.write(self._generate_influence_analysis())
            f.write("\n\n")
            
            # Emotional Dynamics
            f.write("## Emotional Dynamics\n\n")
            f.write(self._generate_emotional_analysis())
            f.write("\n\n")
            
            # Key Findings
            f.write("## Key Findings\n\n")
            f.write(self._generate_key_findings())
            f.write("\n\n")
            
            # Appendices
            f.write("## Appendices\n\n")
            f.write(self._generate_appendices())
        
        print(f"Report generated: {report_path}")
        return report_path
    
    def _generate_executive_summary(self) -> str:
        """Generate executive summary section."""
        final_metrics = self.simulation.metrics_history[-1]
        initial_metrics = self.simulation.metrics_history[0]
        
        summary = f"""This report summarizes a social dynamics simulation involving {len(self.simulation.agents)} agents over {len(self.simulation.metrics_history)} rounds of interaction.

### Key Outcomes:
- **Final Polarization**: {final_metrics.overall_polarization:.3f} (initial: {initial_metrics.overall_polarization:.3f})
- **Final Consensus**: {final_metrics.overall_consensus:.3f} (initial: {initial_metrics.overall_consensus:.3f})
- **Total Conversations**: {len(self.simulation.conversations)}
- **Average Conversation Quality**: {np.mean([c.analysis.overall_quality for c in self.simulation.conversations]):.3f}

### Main Findings:
"""
        
        # Analyze polarization trend
        if final_metrics.overall_polarization > initial_metrics.overall_polarization + 0.1:
            summary += "- The population became significantly more polarized during the simulation\n"
        elif final_metrics.overall_polarization < initial_metrics.overall_polarization - 0.1:
            summary += "- The population became less polarized, moving toward consensus\n"
        else:
            summary += "- Polarization levels remained relatively stable throughout the simulation\n"
        
        # Analyze emotional trends
        if final_metrics.avg_emotional_valence < initial_metrics.avg_emotional_valence - 5:
            summary += "- Average emotional state declined significantly\n"
        elif final_metrics.avg_emotional_valence > initial_metrics.avg_emotional_valence + 5:
            summary += "- Average emotional state improved significantly\n"
        
        # Identify echo chambers
        echo_chambers = self.simulation.analyzer.find_echo_chambers(
            self.simulation.agents, self.simulation.conversations)
        if echo_chambers:
            summary += f"- {len(echo_chambers)} echo chambers formed among agents\n"
        
        return summary
    
    def _generate_parameters_section(self) -> str:
        """Generate simulation parameters section."""
        topics = list(self.simulation.agents.values())[0].opinions.keys()
        
        params = f"""### Basic Configuration:
- **Number of Agents**: {len(self.simulation.agents)}
- **Simulation Rounds**: {len(self.simulation.metrics_history)}
- **Topics Discussed**: {', '.join([t.replace('_', ' ').title() for t in topics])}

### Agent Generation:
- **Personality Model**: HEXACO (6 traits)
- **Opinion Dimensions**: Position, Certainty, Importance, Knowledge, Emotional Charge

### Interaction Settings:
- **Total Conversations**: {len(self.simulation.conversations)}
- **Average Conversations per Round**: {len(self.simulation.conversations) / max(1, len(self.simulation.metrics_history) - 1):.1f}
"""
        return params
    
    def _generate_population_overview(self) -> str:
        """Generate population overview section."""
        # Demographics
        ages = [agent.background.age for agent in self.simulation.agents.values()]
        education_dist = {}
        for agent in self.simulation.agents.values():
            edu = agent.background.education_level
            education_dist[edu] = education_dist.get(edu, 0) + 1
        
        # Personality distribution
        personality_stats = {}
        traits = ['honesty_humility', 'emotionality', 'extraversion',
                 'agreeableness', 'conscientiousness', 'openness']
        
        for trait in traits:
            values = [getattr(agent.personality, trait) 
                     for agent in self.simulation.agents.values()]
            personality_stats[trait] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        overview = f"""### Demographics:
- **Age Range**: {min(ages)} - {max(ages)} (mean: {np.mean(ages):.1f})
- **Education Distribution**:
"""
        
        for edu, count in sorted(education_dist.items()):
            percentage = count / len(self.simulation.agents) * 100
            overview += f"  - {edu}: {count} ({percentage:.1f}%)\n"
        
        overview += "\n### Personality Distribution:\n"
        overview += "| Trait | Mean | Std Dev | Range |\n"
        overview += "|-------|------|---------|-------|\n"
        
        for trait, stats in personality_stats.items():
            trait_name = trait.replace('_', ' ').title()
            overview += f"| {trait_name} | {stats['mean']:.1f} | {stats['std']:.1f} | {stats['min']:.0f}-{stats['max']:.0f} |\n"
        
        return overview
    
    def _generate_opinion_results(self) -> str:
        """Generate opinion dynamics results."""
        topics = list(self.simulation.agents.values())[0].opinions.keys()
        results = "### Opinion Evolution by Topic:\n\n"
        
        for topic in topics:
            initial_metrics = self.simulation.metrics_history[0].opinion_metrics.get(topic, {})
            final_metrics = self.simulation.metrics_history[-1].opinion_metrics.get(topic, {})
            
            results += f"#### {topic.replace('_', ' ').title()}\n"
            results += f"- **Initial Mean Position**: {initial_metrics.get('mean_position', 0):.1f}\n"
            results += f"- **Final Mean Position**: {final_metrics.get('mean_position', 0):.1f}\n"
            results += f"- **Position Change**: {final_metrics.get('mean_position', 0) - initial_metrics.get('mean_position', 0):+.1f}\n"
            results += f"- **Final Polarization**: {final_metrics.get('polarization', 0):.3f}\n"
            results += f"- **Opinion Clusters**: {final_metrics.get('opinion_clusters', 1)}\n\n"
        
        # Overall trends
        results += "### Overall Opinion Trends:\n"
        
        polarization_trend = (self.simulation.metrics_history[-1].overall_polarization - 
                            self.simulation.metrics_history[0].overall_polarization)
        
        if abs(polarization_trend) < 0.05:
            results += "- Polarization remained relatively stable\n"
        elif polarization_trend > 0:
            results += f"- Polarization increased by {polarization_trend:.3f}\n"
        else:
            results += f"- Polarization decreased by {abs(polarization_trend):.3f}\n"
        
        certainty_trend = (self.simulation.metrics_history[-1].avg_certainty - 
                         self.simulation.metrics_history[0].avg_certainty)
        
        if certainty_trend > 5:
            results += f"- Average certainty increased by {certainty_trend:.1f} points\n"
        elif certainty_trend < -5:
            results += f"- Average certainty decreased by {abs(certainty_trend):.1f} points\n"
        
        return results
    
    def _generate_interaction_analysis(self) -> str:
        """Generate interaction analysis section."""
        total_convs = len(self.simulation.conversations)
        
        # Quality distribution
        qualities = [c.analysis.overall_quality for c in self.simulation.conversations]
        
        # Conversation lengths
        lengths = [c.duration_turns for c in self.simulation.conversations]
        
        # Most active agents
        interaction_counts = {}
        for conv in self.simulation.conversations:
            for participant_id in conv.participants:
                interaction_counts[participant_id] = interaction_counts.get(participant_id, 0) + 1
        
        top_active = sorted(interaction_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Handle case with no conversations
        if total_convs == 0:
            analysis = """### Conversation Statistics:
- **Total Conversations**: 0
- **Average Quality**: N/A
- **Average Length**: N/A
- **Quality Range**: N/A

### Most Active Agents:
No conversations occurred in this simulation."""
            return analysis
        
        analysis = f"""### Conversation Statistics:
- **Total Conversations**: {total_convs}
- **Average Quality**: {np.mean(qualities):.3f} (std: {np.std(qualities):.3f})
- **Average Length**: {np.mean(lengths):.1f} turns
- **Quality Range**: {min(qualities):.3f} - {max(qualities):.3f}

### Most Active Agents:
"""
        
        for agent_id, count in top_active:
            agent_name = self.simulation.agents[agent_id].name
            percentage = count / total_convs * 100
            analysis += f"- {agent_name}: {count} conversations ({percentage:.1f}% involvement)\n"
        
        # Conversation patterns
        analysis += "\n### Conversation Patterns:\n"
        
        high_quality = sum(1 for q in qualities if q > 0.7)
        low_quality = sum(1 for q in qualities if q < 0.3)
        
        analysis += f"- High quality conversations (>0.7): {high_quality} ({high_quality/total_convs*100:.1f}%)\n"
        analysis += f"- Low quality conversations (<0.3): {low_quality} ({low_quality/total_convs*100:.1f}%)\n"
        
        return analysis
    
    def _generate_influence_analysis(self) -> str:
        """Generate influence analysis section."""
        # Get influencers
        influencers = self.simulation.analyzer.identify_influencers(
            self.simulation.conversations, self.simulation.agents)[:10]
        
        analysis = "### Top Influencers:\n"
        analysis += "| Rank | Agent | Influence Score | Key Traits |\n"
        analysis += "|------|-------|----------------|------------|\n"
        
        for rank, (name, score) in enumerate(influencers, 1):
            # Find agent
            agent = None
            for a in self.simulation.agents.values():
                if a.name == name:
                    agent = a
                    break
            
            if agent:
                # Get dominant traits
                traits = []
                if agent.personality.extraversion > 70:
                    traits.append("Extraverted")
                if agent.personality.openness > 70:
                    traits.append("Open")
                if agent.personality.conscientiousness > 70:
                    traits.append("Conscientious")
                
                traits_str = ", ".join(traits) if traits else "Balanced"
                analysis += f"| {rank} | {name[:20]} | {score:.1f} | {traits_str} |\n"
        
        # Influence patterns
        analysis += "\n### Influence Patterns:\n"
        
        # Count total opinion changes
        total_changes = 0
        significant_changes = 0
        
        for conv in self.simulation.conversations:
            for agent_id, changes in conv.state_changes.items():
                for topic, deltas in changes.opinion_changes.items():
                    if 'position' in deltas:
                        total_changes += 1
                        if abs(deltas['position']) > 10:
                            significant_changes += 1
        
        analysis += f"- Total opinion changes: {total_changes}\n"
        analysis += f"- Significant changes (>10 points): {significant_changes}\n"
        
        if total_changes > 0:
            analysis += f"- Percentage of significant changes: {significant_changes/total_changes*100:.1f}%\n"
        
        return analysis
    
    def _generate_emotional_analysis(self) -> str:
        """Generate emotional dynamics analysis."""
        # Track emotional changes
        valence_changes = []
        anxiety_changes = []
        
        for conv in self.simulation.conversations:
            for agent_id, changes in conv.state_changes.items():
                if 'valence' in changes.emotion_changes:
                    valence_changes.append(changes.emotion_changes['valence'])
                if 'anxiety' in changes.emotion_changes:
                    anxiety_changes.append(changes.emotion_changes['anxiety'])
        
        # Current emotional state
        current_valences = [agent.emotional_state.valence 
                          for agent in self.simulation.agents.values()]
        current_anxieties = [agent.emotional_state.anxiety 
                           for agent in self.simulation.agents.values()]
        
        analysis = f"""### Population Emotional State:
- **Average Valence**: {np.mean(current_valences):.1f} (range: -50 to +50)
- **Average Anxiety**: {np.mean(current_anxieties):.1f} (range: 0 to 100)
- **Valence Std Dev**: {np.std(current_valences):.1f}
- **Anxiety Std Dev**: {np.std(current_anxieties):.1f}

### Emotional Changes from Interactions:
"""
        
        if valence_changes:
            analysis += f"- **Average Valence Change**: {np.mean(valence_changes):+.2f}\n"
            analysis += f"- **Positive Mood Shifts**: {sum(1 for v in valence_changes if v > 0)}\n"
            analysis += f"- **Negative Mood Shifts**: {sum(1 for v in valence_changes if v < 0)}\n"
        
        if anxiety_changes:
            analysis += f"- **Average Anxiety Change**: {np.mean(anxiety_changes):+.2f}\n"
            analysis += f"- **Anxiety Increases**: {sum(1 for a in anxiety_changes if a > 0)}\n"
            analysis += f"- **Anxiety Decreases**: {sum(1 for a in anxiety_changes if a < 0)}\n"
        
        # Emotional patterns
        analysis += "\n### Emotional Patterns:\n"
        
        # High anxiety agents
        high_anxiety = sum(1 for a in current_anxieties if a > 70)
        analysis += f"- Agents with high anxiety (>70): {high_anxiety} ({high_anxiety/len(self.simulation.agents)*100:.1f}%)\n"
        
        # Negative mood agents
        negative_mood = sum(1 for v in current_valences if v < -20)
        analysis += f"- Agents with negative mood (<-20): {negative_mood} ({negative_mood/len(self.simulation.agents)*100:.1f}%)\n"
        
        return analysis
    
    def _generate_key_findings(self) -> str:
        """Generate key findings section."""
        findings = []
        
        # Check for polarization
        final_pol = self.simulation.metrics_history[-1].overall_polarization
        if final_pol > 0.7:
            findings.append("**High Polarization**: The population has become highly polarized, "
                          "with agents clustering into distinct opinion groups.")
        
        # Check for consensus
        final_cons = self.simulation.metrics_history[-1].overall_consensus
        if final_cons > 0.8:
            findings.append("**Strong Consensus**: Despite initial differences, agents have "
                          "converged toward similar opinions on key topics.")
        
        # Check for echo chambers
        echo_chambers = self.simulation.analyzer.find_echo_chambers(
            self.simulation.agents, self.simulation.conversations)
        if echo_chambers:
            findings.append(f"**Echo Chamber Formation**: {len(echo_chambers)} distinct echo "
                          "chambers formed, where agents primarily interact with similar others.")
        
        # Check emotional trends
        avg_valence = np.mean([a.emotional_state.valence for a in self.simulation.agents.values()])
        if avg_valence < -10:
            findings.append("**Negative Emotional Climate**: The overall emotional state of the "
                          "population is significantly negative.")
        
        # Format findings
        if findings:
            return "\n\n".join(f"{i+1}. {finding}" for i, finding in enumerate(findings))
        else:
            return "No significant patterns detected requiring special attention."
    
    def _generate_appendices(self) -> str:
        """Generate appendices with detailed data."""
        appendices = "### A. Simulation Metadata\n"
        appendices += f"- Simulation ID: {id(self.simulation)}\n"
        appendices += f"- Completion Time: {datetime.now()}\n"
        appendices += f"- Total Agents: {len(self.simulation.agents)}\n"
        appendices += f"- Total Conversations: {len(self.simulation.conversations)}\n"
        
        appendices += "\n### B. Data Export Locations\n"
        appendices += "- `data/agents.json`: Complete agent profiles with biographies and final states\n"
        appendices += "- `data/metrics_history.json`: Population metrics over time\n"
        appendices += "- `data/conversations.json`: Full conversation transcripts and analysis\n"
        appendices += "- `data/analysis.json`: Additional analysis results\n"
        
        return appendices
    
    def generate_csv_exports(self) -> None:
        """Export key data to CSV files for further analysis."""
        # Agent data
        agent_data = []
        for agent in self.simulation.agents.values():
            agent_data.append({
                'id': agent.id,
                'name': agent.name,
                'age': agent.background.age,
                'education': agent.background.education_level,
                'occupation': agent.background.occupation,
                'honesty_humility': agent.personality.honesty_humility,
                'emotionality': agent.personality.emotionality,
                'extraversion': agent.personality.extraversion,
                'agreeableness': agent.personality.agreeableness,
                'conscientiousness': agent.personality.conscientiousness,
                'openness': agent.personality.openness,
                'valence': agent.emotional_state.valence,
                'anxiety': agent.emotional_state.anxiety,
                'confidence': agent.emotional_state.confidence
            })
        
        pd.DataFrame(agent_data).to_csv(
            os.path.join(self.output_dir, 'agents.csv'), index=False)
        
        # Metrics history
        metrics_data = []
        for metric in self.simulation.metrics_history:
            metrics_data.append({
                'round': metric.round_number,
                'polarization': metric.overall_polarization,
                'consensus': metric.overall_consensus,
                'avg_certainty': metric.avg_certainty,
                'avg_valence': metric.avg_emotional_valence,
                'interactions': metric.interaction_count
            })
        
        pd.DataFrame(metrics_data).to_csv(
            os.path.join(self.output_dir, 'metrics_history.csv'), index=False)
        
        print(f"CSV files exported to {self.output_dir}/")
    
    def generate_json_summary(self) -> None:
        """Generate a JSON summary of key results."""
        summary = {
            'metadata': {
                'agents': len(self.simulation.agents),
                'rounds': len(self.simulation.metrics_history),
                'conversations': len(self.simulation.conversations),
                'timestamp': datetime.now().isoformat()
            },
            'final_metrics': {
                'polarization': self.simulation.metrics_history[-1].overall_polarization,
                'consensus': self.simulation.metrics_history[-1].overall_consensus,
                'avg_certainty': self.simulation.metrics_history[-1].avg_certainty,
                'avg_valence': self.simulation.metrics_history[-1].avg_emotional_valence
            },
            'top_influencers': [
                {'name': name, 'score': score}
                for name, score in self.simulation.analyzer.identify_influencers(
                    self.simulation.conversations, self.simulation.agents)[:5]
            ],
            'echo_chambers': len(self.simulation.analyzer.find_echo_chambers(
                self.simulation.agents, self.simulation.conversations))
        }
        
        with open(os.path.join(self.output_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"JSON summary saved to {self.output_dir}/summary.json")