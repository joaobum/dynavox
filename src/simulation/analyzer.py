"""Analysis tools for simulation results."""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np
from datetime import datetime
from ..agents.profile import Agent
from ..interactions.orchestrator import Conversation


@dataclass
class PopulationMetrics:
    """Metrics for the agent population at a point in time."""
    timestamp: datetime
    round_number: int
    
    # Opinion metrics per topic
    opinion_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Overall metrics
    overall_polarization: float = 0.0
    overall_consensus: float = 0.0
    avg_certainty: float = 0.0
    avg_emotional_valence: float = 0.0
    
    # Interaction metrics
    interaction_count: int = 0
    avg_interaction_quality: float = 0.0
    

class SimulationAnalyzer:
    """Analyzes simulation results and calculates metrics."""
    
    def calculate_population_metrics(self, agents: Dict[str, Agent], 
                                   conversations: List[Conversation],
                                   round_number: int) -> PopulationMetrics:
        """Calculate comprehensive metrics for the population."""
        metrics = PopulationMetrics(
            timestamp=datetime.now(),
            round_number=round_number
        )
        
        if not agents:
            return metrics
        
        # Get all topics from first agent (assuming all have same topics)
        topics = list(next(iter(agents.values())).opinions.keys())
        
        # Calculate per-topic metrics
        for topic in topics:
            topic_metrics = self._calculate_topic_metrics(agents, topic)
            metrics.opinion_metrics[topic] = topic_metrics
        
        # Calculate overall metrics
        metrics.overall_polarization = self._calculate_overall_polarization(
            metrics.opinion_metrics)
        metrics.overall_consensus = self._calculate_overall_consensus(
            metrics.opinion_metrics)
        metrics.avg_certainty = self._calculate_average_certainty(agents)
        metrics.avg_emotional_valence = self._calculate_average_valence(agents)
        
        # Interaction metrics for this round
        round_conversations = [c for c in conversations 
                             if c.timestamp.date() == datetime.now().date()]
        metrics.interaction_count = len(round_conversations)
        
        if round_conversations:
            qualities = [c.analysis.overall_quality for c in round_conversations]
            metrics.avg_interaction_quality = np.mean(qualities)
        
        return metrics
    
    def _calculate_topic_metrics(self, agents: Dict[str, Agent], 
                               topic: str) -> Dict[str, float]:
        """Calculate metrics for a specific topic."""
        positions = []
        certainties = []
        importances = []
        agent_ids = []
        
        for agent_id, agent in agents.items():
            if topic in agent.opinions:
                positions.append(agent.opinions[topic].position)
                certainties.append(agent.opinions[topic].certainty)
                importances.append(agent.opinions[topic].importance)
                agent_ids.append(agent_id)
        
        if not positions:
            return {}
        
        positions_array = np.array(positions)
        
        # Get cluster assignments and summaries
        cluster_info = self._identify_clusters_detailed(positions_array, agent_ids, agents, topic)
        
        return {
            'mean_position': float(np.mean(positions_array)),
            'std_position': float(np.std(positions_array)),
            'polarization': self._calculate_polarization(positions_array),
            'consensus': self._calculate_consensus(positions_array),
            'mean_certainty': float(np.mean(certainties)),
            'mean_importance': float(np.mean(importances)),
            'opinion_clusters': cluster_info['num_clusters'],
            'cluster_summaries': cluster_info['summaries']
        }
    
    def _calculate_polarization(self, positions: np.ndarray) -> float:
        """Calculate polarization as bimodality of opinion distribution."""
        if len(positions) < 10:
            # Not enough data for meaningful polarization
            return 0.0
        
        # Create histogram
        hist, bin_edges = np.histogram(positions, bins=20, range=(-100, 100))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Find peaks (local maxima)
        peaks = []
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > 0:
                peaks.append((bin_centers[i], hist[i]))
        
        if len(peaks) < 2:
            return 0.0
        
        # Sort peaks by height
        peaks.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate polarization based on distance between two highest peaks
        if len(peaks) >= 2:
            peak1_pos, peak1_height = peaks[0]
            peak2_pos, peak2_height = peaks[1]
            
            # Distance normalized to [-100, 100] range
            distance = abs(peak1_pos - peak2_pos) / 200
            
            # Weight by relative heights of peaks
            height_ratio = min(peak1_height, peak2_height) / max(peak1_height, peak2_height)
            
            # Polarization combines distance and balance of peaks
            polarization = distance * height_ratio
            
            return float(min(1.0, polarization))
        
        return 0.0
    
    def _calculate_consensus(self, positions: np.ndarray) -> float:
        """Calculate consensus as inverse of opinion spread."""
        if len(positions) < 2:
            return 1.0
        
        # Use standard deviation as measure of spread
        std = np.std(positions)
        
        # Normalize to [0, 1] where 1 is perfect consensus
        # Max possible std is 100 (from -100 to 100)
        consensus = 1 - (std / 100)
        
        return float(max(0.0, consensus))
    
    def _identify_clusters(self, positions: np.ndarray) -> int:
        """Identify number of opinion clusters using simple method."""
        if len(positions) < 5:
            return 1
        
        # Sort positions
        sorted_pos = np.sort(positions)
        
        # Find gaps larger than threshold
        threshold = 30  # Gap of 30 points indicates separate cluster
        clusters = 1
        
        for i in range(1, len(sorted_pos)):
            if sorted_pos[i] - sorted_pos[i-1] > threshold:
                clusters += 1
        
        return min(clusters, 5)  # Cap at 5 clusters
    
    def _identify_clusters_detailed(self, positions: np.ndarray, agent_ids: List[str],
                                  agents: Dict[str, Agent], topic: str) -> Dict:
        """Identify opinion clusters with detailed summaries."""
        if len(positions) < 5:
            # Single cluster
            return {
                'num_clusters': 1,
                'summaries': [{
                    'cluster_id': 0,
                    'size': len(positions),
                    'mean_position': float(np.mean(positions)),
                    'position_range': (float(np.min(positions)), float(np.max(positions))),
                    'description': self._describe_cluster_position(float(np.mean(positions)), topic),
                    'agent_count': len(positions),
                    'percentage': 100.0  # Single cluster contains all agents
                }]
            }
        
        # Sort positions with agent IDs
        sorted_indices = np.argsort(positions)
        sorted_positions = positions[sorted_indices]
        sorted_agent_ids = [agent_ids[i] for i in sorted_indices]
        
        # Find cluster boundaries
        threshold = 30
        cluster_boundaries = [0]
        
        for i in range(1, len(sorted_positions)):
            if sorted_positions[i] - sorted_positions[i-1] > threshold:
                cluster_boundaries.append(i)
        cluster_boundaries.append(len(sorted_positions))
        
        # Create cluster summaries
        summaries = []
        for i in range(len(cluster_boundaries) - 1):
            start_idx = cluster_boundaries[i]
            end_idx = cluster_boundaries[i + 1]
            
            cluster_positions = sorted_positions[start_idx:end_idx]
            cluster_agent_ids = sorted_agent_ids[start_idx:end_idx]
            
            # Calculate cluster statistics
            mean_pos = float(np.mean(cluster_positions))
            
            summary = {
                'cluster_id': i,
                'size': len(cluster_positions),
                'mean_position': mean_pos,
                'position_range': (float(np.min(cluster_positions)), 
                                 float(np.max(cluster_positions))),
                'description': self._describe_cluster_position(mean_pos, topic),
                'agent_count': len(cluster_positions),
                'percentage': len(cluster_positions) / len(positions) * 100,
                # Sample agent characteristics (first 3)
                'sample_agents': [
                    {
                        'name': agents[aid].name,
                        'occupation': agents[aid].background.occupation,
                        'age': agents[aid].background.age
                    }
                    for aid in cluster_agent_ids[:3]
                ]
            }
            summaries.append(summary)
        
        return {
            'num_clusters': len(summaries),
            'summaries': summaries
        }
    
    def _describe_cluster_position(self, mean_position: float, topic: str) -> str:
        """Generate human-readable description of cluster position."""
        topic_display = topic.replace('_', ' ')
        
        if mean_position > 50:
            stance = "strongly support"
        elif mean_position > 20:
            stance = "moderately support"
        elif mean_position > -20:
            stance = "are neutral about"
        elif mean_position > -50:
            stance = "moderately oppose"
        else:
            stance = "strongly oppose"
        
        return f"Agents who {stance} {topic_display}"
    
    def _calculate_overall_polarization(self, 
                                      opinion_metrics: Dict[str, Dict[str, float]]) -> float:
        """Calculate overall polarization across all topics."""
        polarizations = [
            metrics.get('polarization', 0.0) 
            for metrics in opinion_metrics.values()
        ]
        
        return float(np.mean(polarizations)) if polarizations else 0.0
    
    def _calculate_overall_consensus(self,
                                   opinion_metrics: Dict[str, Dict[str, float]]) -> float:
        """Calculate overall consensus across all topics."""
        consensuses = [
            metrics.get('consensus', 0.0)
            for metrics in opinion_metrics.values()
        ]
        
        return float(np.mean(consensuses)) if consensuses else 0.0
    
    def _calculate_average_certainty(self, agents: Dict[str, Agent]) -> float:
        """Calculate average opinion certainty across all agents and topics."""
        certainties = []
        
        for agent in agents.values():
            for opinion in agent.opinions.values():
                certainties.append(opinion.certainty)
        
        return float(np.mean(certainties)) if certainties else 0.0
    
    def _calculate_average_valence(self, agents: Dict[str, Agent]) -> float:
        """Calculate average emotional valence across all agents."""
        valences = [agent.emotional_state.valence for agent in agents.values()]
        return float(np.mean(valences)) if valences else 0.0
    
    def identify_influencers(self, conversations: List[Conversation],
                           agents: Dict[str, Agent]) -> List[Tuple[str, float]]:
        """Identify agents who had the most influence on others."""
        influence_scores = {}
        
        for conversation in conversations:
            # Look at state changes
            for agent_id, changes in conversation.state_changes.items():
                # The OTHER agent in the conversation influenced this one
                partner_id = [pid for pid in conversation.participants 
                            if pid != agent_id][0]
                
                if partner_id not in influence_scores:
                    influence_scores[partner_id] = 0.0
                
                # Sum absolute opinion changes caused
                total_change = 0.0
                for topic_changes in changes.opinion_changes.values():
                    total_change += abs(topic_changes.get('position', 0))
                
                influence_scores[partner_id] += total_change
        
        # Sort by influence score
        sorted_influencers = sorted(influence_scores.items(), 
                                  key=lambda x: x[1], reverse=True)
        
        # Return agent names with scores
        return [(agents[aid].name, score) for aid, score in sorted_influencers[:10]
                if aid in agents]
    
    def analyze_opinion_trajectories(self, metrics_history: List[PopulationMetrics],
                                   topic: str) -> Dict[str, List[float]]:
        """Analyze how opinions on a topic evolved over time."""
        trajectories = {
            'mean_positions': [],
            'std_positions': [],
            'polarizations': [],
            'consensuses': []
        }
        
        for metrics in metrics_history:
            if topic in metrics.opinion_metrics:
                topic_data = metrics.opinion_metrics[topic]
                trajectories['mean_positions'].append(
                    topic_data.get('mean_position', 0))
                trajectories['std_positions'].append(
                    topic_data.get('std_position', 0))
                trajectories['polarizations'].append(
                    topic_data.get('polarization', 0))
                trajectories['consensuses'].append(
                    topic_data.get('consensus', 0))
        
        return trajectories
    
    def find_echo_chambers(self, agents: Dict[str, Agent],
                         conversations: List[Conversation]) -> List[List[str]]:
        """Identify groups of agents who primarily interact with similar others."""
        # Build interaction graph
        interaction_counts = {}
        
        for conv in conversations:
            pair = tuple(sorted(conv.participants))
            interaction_counts[pair] = interaction_counts.get(pair, 0) + 1
        
        # Find agents who repeatedly interact
        frequent_pairs = [
            pair for pair, count in interaction_counts.items() 
            if count >= 3  # At least 3 interactions
        ]
        
        # Group into clusters
        clusters = []
        for pair in frequent_pairs:
            agent1_id, agent2_id = pair
            
            # Check opinion similarity
            if agent1_id in agents and agent2_id in agents:
                agent1 = agents[agent1_id]
                agent2 = agents[agent2_id]
                
                # Calculate opinion similarity
                similarity = self._calculate_opinion_similarity(agent1, agent2)
                
                if similarity > 0.7:  # High similarity threshold
                    # Add to existing cluster or create new one
                    added = False
                    for cluster in clusters:
                        if agent1_id in cluster or agent2_id in cluster:
                            cluster.add(agent1_id)
                            cluster.add(agent2_id)
                            added = True
                            break
                    
                    if not added:
                        clusters.append({agent1_id, agent2_id})
        
        # Convert to lists of agent names
        echo_chambers = []
        for cluster in clusters:
            if len(cluster) >= 3:  # At least 3 agents
                names = [agents[aid].name for aid in cluster if aid in agents]
                echo_chambers.append(names)
        
        return echo_chambers
    
    def _calculate_opinion_similarity(self, agent1: Agent, agent2: Agent) -> float:
        """Calculate similarity between two agents' opinions."""
        shared_topics = set(agent1.opinions.keys()) & set(agent2.opinions.keys())
        
        if not shared_topics:
            return 0.0
        
        distances = []
        for topic in shared_topics:
            # Position distance (normalized)
            pos_dist = abs(agent1.opinions[topic].position - 
                          agent2.opinions[topic].position) / 200
            distances.append(pos_dist)
        
        # Similarity is inverse of average distance
        avg_distance = np.mean(distances)
        return 1 - avg_distance
    
    def generate_cluster_report(self, metrics_history: List[PopulationMetrics]) -> Dict:
        """Generate a comprehensive report of opinion clusters over time."""
        report = {}
        
        # Get all topics from the latest metrics
        if not metrics_history:
            return report
        
        latest_metrics = metrics_history[-1]
        topics = list(latest_metrics.opinion_metrics.keys())
        
        for topic in topics:
            topic_report = {
                'evolution': [],
                'final_state': None
            }
            
            # Track cluster evolution
            for metrics in metrics_history:
                if topic in metrics.opinion_metrics:
                    topic_data = metrics.opinion_metrics[topic]
                    round_summary = {
                        'round': metrics.round_number,
                        'num_clusters': topic_data.get('opinion_clusters', 1),
                        'cluster_summaries': topic_data.get('cluster_summaries', [])
                    }
                    topic_report['evolution'].append(round_summary)
            
            # Final state summary
            if topic in latest_metrics.opinion_metrics:
                final_data = latest_metrics.opinion_metrics[topic]
                topic_report['final_state'] = {
                    'num_clusters': final_data.get('opinion_clusters', 1),
                    'polarization': final_data.get('polarization', 0),
                    'consensus': final_data.get('consensus', 0),
                    'cluster_details': final_data.get('cluster_summaries', [])
                }
            
            report[topic] = topic_report
        
        return report