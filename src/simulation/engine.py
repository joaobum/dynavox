"""Main simulation engine for multi-agent social dynamics."""
import random
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import os
from ..agents.profile import Agent
from ..agents.generator import AgentGenerator
from ..interactions.orchestrator import ConversationOrchestrator, Conversation
from ..llm.client import LLMClient
from .analyzer import SimulationAnalyzer, PopulationMetrics

logger = logging.getLogger('dynavox.simulation')


class SimulationEngine:
    """Main simulation coordinator for agent-based social dynamics."""
    
    def __init__(self, llm_client: LLMClient, seed: Optional[int] = None, use_async: bool = False):
        """Initialize simulation engine.
        
        Args:
            llm_client: LLM client for agent generation and conversations
            seed: Random seed for reproducibility
            use_async: Whether to use async conversation execution
        """
        logger.debug(f"Initializing SimulationEngine with seed={seed}, use_async={use_async}")
        self.llm_client = llm_client
        self.generator = AgentGenerator(llm_client)
        self.orchestrator = ConversationOrchestrator(llm_client)
        self.async_orchestrator = None
        self.use_async = use_async
        
        if use_async:
            from ..interactions.async_orchestrator import AsyncConversationOrchestrator
            self.async_orchestrator = AsyncConversationOrchestrator(llm_client)
            logger.info("Async conversation mode enabled for parallel interactions")
            logger.debug("Async conversation mode enabled for parallel interactions")
        
        self.analyzer = SimulationAnalyzer()
        
        # Log the model being used if available
        if hasattr(llm_client, 'model'):
            logger.info(f"Simulation engine initialized with model: {llm_client.model}")
            logger.debug(f"Simulation engine initialized with model: {llm_client.model}")
        
        # Simulation state
        self.agents: Dict[str, Agent] = {}
        self.conversations: List[Conversation] = []
        self.metrics_history: List[PopulationMetrics] = []
        self.round_number = 0
        
        # Real-time data writer (will be set by QuickSimulation)
        self.data_writer = None
        
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            logger.debug(f"Random seed set to {seed}")
    
    def initialize_population(self, size: int, topics: List[str], 
                            demographics: Optional[Dict] = None,
                            personality_distribution: Optional[Dict] = None) -> None:
        """Create initial agent population.
        
        Args:
            size: Number of agents to create
            topics: List of opinion topics for agents
            demographics: Optional demographic constraints
            personality_distribution: Optional personality trait distributions
        """
        logger.info(f"Initializing population of {size} agents with topics: {topics}")
        logger.debug(f"Initializing population of {size} agents...")
        
        for i in range(size):
            # Build constraints for this agent
            constraints = {
                'topics': topics,
                'demographics': demographics or {}
            }
            
            # Add personality bias for diversity
            if personality_distribution:
                # Use provided distribution
                constraints['personality_bias'] = self._sample_personality_bias(
                    personality_distribution, i)
            else:
                # Default diverse distribution - more variety
                biases = [
                    'high openness',
                    'high conscientiousness', 
                    'high agreeableness',
                    'high extraversion',
                    'balanced',
                    'low agreeableness',
                    'high emotionality',
                    'low openness',
                    'low conscientiousness',
                    'low extraversion',
                    'high honesty_humility',
                    'low honesty_humility',
                    'mixed_introverted',  # Low extraversion, high conscientiousness
                    'mixed_creative',     # High openness, low conscientiousness
                    'mixed_anxious',      # High emotionality, low extraversion
                    'mixed_leader',       # High extraversion, high conscientiousness
                    'mixed_rebel'         # Low agreeableness, high openness
                ]
                # Use modulo but also add some randomness for larger populations
                bias_index = i % len(biases)
                if i >= len(biases) and random.random() < 0.3:
                    # 30% chance to pick a random bias for variety
                    bias_index = random.randint(0, len(biases) - 1)
                constraints['personality_bias'] = biases[bias_index]
            
            # Add diverse demographic constraints
            constraints['demographics'] = demographics or self._generate_diverse_demographics(i, size)
            
            try:
                logger.debug(f"Generating agent {i+1} with constraints: {constraints}")
                agent = self.generator.generate_agent(constraints)
                self.agents[agent.id] = agent
                logger.info(f"👤 Created agent {agent.id}: {agent.name} ({agent.background.age}, {agent.background.occupation})")
                # Agent creation already logged above
                
                # Update real-time data if writer is available
                if self.data_writer:
                    self.data_writer.update_agent(agent)
            except Exception as e:
                logger.error(f"Failed to create agent {i+1}: {e}")
                # Error already logged above
                # Try again with fewer constraints
                try:
                    logger.debug("Retrying with default constraints")
                    agent = self.generator.generate_agent({'topics': topics})
                    self.agents[agent.id] = agent
                    logger.info(f"Created agent {agent.id} with default constraints")
                    # Agent creation already logged above
                except:
                    logger.warning(f"Skipping agent {i+1} due to repeated failures")
        
        logger.info(f"✅ Successfully created {len(self.agents)} agents")
        
        # Calculate initial metrics
        initial_metrics = self.analyzer.calculate_population_metrics(
            self.agents, [], 0)
        self.metrics_history.append(initial_metrics)
    
    def run_interaction_round(self, 
                            interaction_probability: float = 0.1,
                            homophily_bias: float = 0.5,
                            max_interactions_per_agent: int = 2) -> None:
        """Run one round of interactions between agents.
        
        Creates n/2 potential interaction pairs, then evaluates each pair
        for actual interaction based on probability and other factors.
        All accepted interactions happen concurrently using async.
        
        Args:
            interaction_probability: Base probability of any two agents interacting
            homophily_bias: Strength of preference for similar agents (0-1)
            max_interactions_per_agent: Maximum interactions per agent per round
        """
        self.round_number += 1
        logger.info(f"🔁 Starting interaction round {self.round_number}")
        # Round already logged above
        
        agent_list = list(self.agents.values())
        n_agents = len(agent_list)
        target_pairs = n_agents // 2
        
        # Shuffle agents for random pairing
        shuffled_agents = agent_list.copy()
        random.shuffle(shuffled_agents)
        
        # Create n/2 potential pairs
        potential_pairs = []
        used_agents = set()
        
        # First, try to create pairs from shuffled list
        for i in range(0, len(shuffled_agents) - 1, 2):
            if len(potential_pairs) >= target_pairs:
                break
            agent1 = shuffled_agents[i]
            agent2 = shuffled_agents[i + 1]
            potential_pairs.append((agent1, agent2))
            used_agents.add(agent1.id)
            used_agents.add(agent2.id)
        
        # If we need more pairs, create from remaining agents
        remaining_agents = [a for a in agent_list if a.id not in used_agents]
        while len(potential_pairs) < target_pairs and len(remaining_agents) >= 2:
            agent1 = remaining_agents.pop(random.randint(0, len(remaining_agents) - 1))
            agent2 = remaining_agents.pop(random.randint(0, len(remaining_agents) - 1))
            potential_pairs.append((agent1, agent2))
        
        logger.debug(f"🤝 Formed {len(potential_pairs)} potential conversation pairs")
        logger.info(f"🤝 Formed {len(potential_pairs)} potential conversation pairs")
        
        # Evaluate each pair for actual interaction
        interaction_pairs = []
        for agent1, agent2 in potential_pairs:
            # Calculate interaction probability
            base_prob = interaction_probability
            
            # Apply homophily bias
            similarity = self._calculate_similarity(agent1, agent2)
            adjusted_prob = base_prob * (1 + homophily_bias * similarity)
            
            # Apply social energy modifier
            energy_modifier = (agent1.emotional_state.social_energy + 
                             agent2.emotional_state.social_energy) / 200
            adjusted_prob *= energy_modifier
            
            # Decision to interact
            if random.random() < adjusted_prob:
                interaction_pairs.append((agent1, agent2))
                logger.debug(f"✅ {agent1.name} and {agent2.name} will interact (prob: {adjusted_prob:.2f})")
                logger.debug(f"  ✅ {agent1.name} and {agent2.name} will interact (prob: {adjusted_prob:.2f})")
            else:
                logger.debug(f"❌ {agent1.name} and {agent2.name} decided not to interact (prob: {adjusted_prob:.2f})")
                logger.debug(f"  ❌ {agent1.name} and {agent2.name} decided not to interact (prob: {adjusted_prob:.2f})")
        
        # Execute conversations
        if self.use_async and self.async_orchestrator and interaction_pairs:
            logger.info(f"Executing {len(interaction_pairs)} conversations in parallel")
            # Parallel execution already logged below
            # Async parallel execution
            import asyncio
            
            async def run_parallel_conversations():
                conversations = await self.async_orchestrator.conduct_conversations_parallel(
                    interaction_pairs)
                return conversations
            
            # Execute async conversations
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                new_conversations = loop.run_until_complete(run_parallel_conversations())
                loop.close()
                
                # Add to conversation history
                self.conversations.extend(new_conversations)
                logger.info(f"Completed {len(new_conversations)} parallel conversations")
                # Completion already logged above
                
                # Print all conversation summaries after parallel completion
                logger.info("📋 Conversation Summaries:")
                for i, conv in enumerate(new_conversations):
                    agent1 = self.agents[conv.participants[0]]
                    agent2 = self.agents[conv.participants[1]]
                    self.orchestrator._print_conversation_summary(agent1, agent2, conv)
                
            except Exception as e:
                logger.error(f"Error in parallel conversation execution: {e}")
                logger.error(f"Async conversation execution failed: {e}")
                logger.warning("Falling back to synchronous execution...")
                self.use_async = False
        
        if not self.use_async and interaction_pairs:
            # Synchronous execution
            logger.info(f"Executing {len(interaction_pairs)} conversations sequentially")
            # Sequential execution already logged below
            for agent1, agent2 in interaction_pairs:
                try:
                    logger.debug(f"Starting conversation between {agent1.name} and {agent2.name}")
                    logger.debug(f"  {agent1.name} is talking with {agent2.name}...")
                    
                    conversation = self.orchestrator.conduct_conversation(agent1, agent2)
                    self.conversations.append(conversation)
                    logger.info(f"💬 Conversation completed between {agent1.id} and {agent2.id}")
                    
                    # Update real-time data if writer is available
                    if self.data_writer:
                        self.data_writer.add_conversation(conversation)
                        # Update both agents after conversation
                        self.data_writer.update_agent(agent1)
                        self.data_writer.update_agent(agent2)
                    
                    # Print summary is now done inside conduct_conversation
                    
                except Exception as e:
                    logger.error(f"Conversation failed between {agent1.id} and {agent2.id}: {e}")
                    print(f"    Conversation failed: {e}")
        
        print(f"\nRound {self.round_number} complete")
        
        # Calculate and store metrics
        round_metrics = self.analyzer.calculate_population_metrics(
            self.agents, self.conversations, self.round_number)
        self.metrics_history.append(round_metrics)
        
        # Update real-time metrics if writer is available
        if self.data_writer:
            self.data_writer.update_metrics(round_metrics)
        
        # Print summary metrics
        self._print_round_summary(round_metrics)
    
    def run_simulation(self, rounds: int, **kwargs) -> None:
        """Run complete simulation for specified number of rounds.
        
        Args:
            rounds: Number of rounds to simulate
            **kwargs: Arguments passed to run_interaction_round
        """
        logger.info(f"Starting simulation with {len(self.agents)} agents for {rounds} rounds")
        logger.debug(f"Simulation parameters: {kwargs}")
        print(f"\nStarting simulation with {len(self.agents)} agents for {rounds} rounds")
        
        for round_num in range(rounds):
            self.run_interaction_round(**kwargs)
            
            # Optional: Save checkpoint
            if round_num % 5 == 0 and hasattr(self, 'checkpoint_dir'):
                self.save_checkpoint(f"checkpoint_round_{round_num}", self.checkpoint_dir)
    
    def _calculate_similarity(self, agent1: Agent, agent2: Agent) -> float:
        """Calculate similarity between agents for homophily calculations."""
        logger.debug(f"Calculating similarity between {agent1.name} and {agent2.name}")
        similarity_components = []
        
        # Personality similarity (weighted)
        personality_traits = ['honesty_humility', 'emotionality', 'extraversion',
                            'agreeableness', 'conscientiousness', 'openness']
        
        personality_diffs = []
        for trait in personality_traits:
            val1 = getattr(agent1.personality, trait)
            val2 = getattr(agent2.personality, trait)
            diff = abs(val1 - val2) / 100
            personality_diffs.append(diff)
            logger.debug(f"  {trait}: {val1} vs {val2}, diff={diff:.3f}")
        
        personality_similarity = 1 - sum(personality_diffs) / len(personality_diffs)
        similarity_components.append(('personality', personality_similarity, 0.3))
        logger.debug(f"  Personality similarity: {personality_similarity:.3f}")
        
        # Background similarity
        background_score = 0.0
        
        # Education level
        if agent1.background.education_level == agent2.background.education_level:
            background_score += 0.25
        
        # Age similarity (within 10 years)
        age_diff = abs(agent1.background.age - agent2.background.age)
        if age_diff <= 10:
            background_score += 0.25 * (1 - age_diff / 10)
        
        # Cultural tag overlap
        cultural_overlap = len(set(agent1.background.cultural_tags) & 
                             set(agent2.background.cultural_tags))
        if agent1.background.cultural_tags and agent2.background.cultural_tags:
            cultural_similarity = cultural_overlap / max(
                len(agent1.background.cultural_tags),
                len(agent2.background.cultural_tags))
            background_score += 0.25 * cultural_similarity
        
        # Socioeconomic similarity
        socio_overlap = len(set(agent1.background.socioeconomic_tags) & 
                          set(agent2.background.socioeconomic_tags))
        if agent1.background.socioeconomic_tags and agent2.background.socioeconomic_tags:
            socio_similarity = socio_overlap / max(
                len(agent1.background.socioeconomic_tags),
                len(agent2.background.socioeconomic_tags))
            background_score += 0.25 * socio_similarity
        
        similarity_components.append(('background', background_score, 0.3))
        logger.debug(f"  Background similarity: {background_score:.3f}")
        
        # Opinion similarity
        shared_topics = set(agent1.opinions.keys()) & set(agent2.opinions.keys())
        if shared_topics:
            opinion_diffs = []
            for topic in shared_topics:
                pos1 = agent1.opinions[topic].position
                pos2 = agent2.opinions[topic].position
                pos_diff = abs(pos1 - pos2) / 200
                opinion_diffs.append(pos_diff)
                logger.debug(f"  Opinion on {topic}: {pos1} vs {pos2}, diff={pos_diff:.3f}")
            
            opinion_similarity = 1 - sum(opinion_diffs) / len(opinion_diffs)
            similarity_components.append(('opinions', opinion_similarity, 0.4))
            logger.debug(f"  Opinion similarity: {opinion_similarity:.3f}")
        
        # Calculate weighted total
        total_similarity = sum(score * weight for _, score, weight in similarity_components)
        logger.debug(f"Total similarity between {agent1.name} and {agent2.name}: {total_similarity:.3f}")
        
        return total_similarity
    
    def _sample_personality_bias(self, distribution: Dict, index: int) -> str:
        """Sample personality bias from distribution."""
        # Simple implementation - could be made more sophisticated
        biases = list(distribution.keys())
        weights = list(distribution.values())
        
        # Normalize weights
        total = sum(weights)
        weights = [w/total for w in weights]
        
        # Simple weighted selection
        return random.choices(biases, weights=weights)[0]
    
    def _generate_diverse_demographics(self, agent_index: int, total_size: int) -> Dict:
        """Generate diverse demographic constraints for agent creation."""
        from ..config import AGENT_GENERATION_PARAMS
        
        demographics = {}
        
        # Vary age distribution
        if agent_index % 5 == 0:
            # 20% younger adults
            demographics['age_range'] = (18, 30)
        elif agent_index % 5 == 1:
            # 20% middle aged
            demographics['age_range'] = (31, 45)
        elif agent_index % 5 == 2:
            # 20% mature adults
            demographics['age_range'] = (46, 60)
        elif agent_index % 5 == 3:
            # 20% older adults
            demographics['age_range'] = (61, 75)
        else:
            # 20% elderly
            demographics['age_range'] = (76, 85)
        
        # Education distribution - weight towards lower education
        edu_weights = AGENT_GENERATION_PARAMS['education_distribution']
        education_levels = list(edu_weights.keys())
        weights = list(edu_weights.values())
        demographics['education_preference'] = random.choices(education_levels, weights=weights)[0]
        
        # Socioeconomic distribution
        socio_dist = AGENT_GENERATION_PARAMS['socioeconomic_distribution']
        socio_classes = list(socio_dist.keys())
        socio_weights = list(socio_dist.values())
        demographics['socioeconomic_preference'] = random.choices(socio_classes, weights=socio_weights)[0]
        
        # Occupation category - correlate with education and socioeconomic status
        occupation_map = {
            'no_high_school': ['unemployed', 'service', 'manual_labor'],
            'high_school': ['service', 'manual_labor', 'clerical', 'skilled_trades'],
            'some_college': ['clerical', 'skilled_trades', 'service', 'education'],
            'associates': ['skilled_trades', 'healthcare', 'clerical', 'business'],
            'bachelors': ['business', 'education', 'healthcare', 'professional', 'creative'],
            'masters': ['professional', 'business', 'education', 'healthcare'],
            'phd': ['professional', 'education', 'healthcare']
        }
        
        # Pick occupation category based on education
        edu_level = demographics['education_preference']
        possible_occupations = occupation_map.get(edu_level, ['service', 'clerical'])
        
        # Add some randomness - 20% chance of unexpected occupation
        if random.random() < 0.2:
            all_categories = list(AGENT_GENERATION_PARAMS['occupation_categories'].keys())
            demographics['occupation_category'] = random.choice(all_categories)
        else:
            demographics['occupation_category'] = random.choice(possible_occupations)
        
        # Location preferences
        location_types = ['rural', 'suburban', 'urban']
        location_weights = [0.25, 0.45, 0.3]  # US-like distribution
        demographics['location_preference'] = random.choices(location_types, location_weights)[0]
        
        # Family/relationship diversity
        if agent_index % 4 == 0:
            demographics['relationship_preference'] = 'single'
        elif agent_index % 4 == 1:
            demographics['relationship_preference'] = 'partnered'
        elif agent_index % 4 == 2:
            demographics['relationship_preference'] = 'family_focused'
        else:
            demographics['relationship_preference'] = 'complex'  # divorced, widowed, etc
        
        return demographics
    
    def _print_round_summary(self, metrics: PopulationMetrics) -> None:
        """Print summary of round metrics."""
        print(f"\nRound {metrics.round_number} Summary:")
        print(f"  Overall polarization: {metrics.overall_polarization:.3f}")
        print(f"  Overall consensus: {metrics.overall_consensus:.3f}")
        print(f"  Average certainty: {metrics.avg_certainty:.1f}")
        print(f"  Average mood: {metrics.avg_emotional_valence:.1f}")
        print(f"  Interaction quality: {metrics.avg_interaction_quality:.3f}")
        
        # Per-topic summary
        for topic, topic_metrics in metrics.opinion_metrics.items():
            print(f"\n  {topic}:")
            print(f"    Mean position: {topic_metrics['mean_position']:.1f}")
            print(f"    Polarization: {topic_metrics['polarization']:.3f}")
            print(f"    Clusters: {topic_metrics['opinion_clusters']}")
            
            # Print cluster summaries if available
            if 'cluster_summaries' in topic_metrics:
                for cluster in topic_metrics['cluster_summaries']:
                    print(f"      Cluster {cluster['cluster_id'] + 1}: "
                          f"{cluster['percentage']:.0f}% of agents "
                          f"({cluster['description']})")
    
    def save_checkpoint(self, filename: str, checkpoint_dir: str = 'checkpoints') -> None:
        """Save simulation state to file."""
        checkpoint = {
            'round_number': self.round_number,
            'timestamp': datetime.now().isoformat(),
            'agents': self._serialize_agents(),
            'metrics_history': self._serialize_metrics(),
            'conversation_count': len(self.conversations),
            'use_async': self.use_async
        }
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        filepath = os.path.join(checkpoint_dir, f'{filename}.json')
        
        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        print(f"Saved checkpoint to {filepath}")
    
    def export_results(self, output_dir: str = 'results') -> None:
        """Export comprehensive simulation results."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create data subdirectory
        data_dir = os.path.join(output_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        # Export agent data with full biographies
        agents_data = {
            agent_id: {
                'id': agent.id,
                'name': agent.name,
                'biography': agent.biography,  # Include full biography
                'conversation_style': agent.conversation_style,  # Include conversation style
                'age': agent.background.age,
                'occupation': agent.background.occupation,
                'education_level': agent.background.education_level,
                'education_field': agent.background.education_field,
                'background_tags': {
                    'socioeconomic': agent.background.socioeconomic_tags,
                    'relationship': agent.background.relationship_tags,
                    'cultural': agent.background.cultural_tags
                },
                'personality': {
                    'honesty_humility': agent.personality.honesty_humility,
                    'emotionality': agent.personality.emotionality,
                    'extraversion': agent.personality.extraversion,
                    'agreeableness': agent.personality.agreeableness,
                    'conscientiousness': agent.personality.conscientiousness,
                    'openness': agent.personality.openness
                },
                'emotional_baseline': {
                    'dispositional_affect': agent.emotional_baseline.dispositional_affect,
                    'stress_tolerance': agent.emotional_baseline.stress_tolerance,
                    'social_confidence': agent.emotional_baseline.social_confidence,
                    'self_efficacy': agent.emotional_baseline.self_efficacy
                },
                'final_opinions': {
                    topic: {
                        'position': op.position,
                        'certainty': op.certainty,
                        'importance': op.importance,
                        'knowledge': op.knowledge,
                        'emotional_charge': op.emotional_charge
                    }
                    for topic, op in agent.opinions.items()
                },
                'final_emotional_state': {
                    'arousal': agent.emotional_state.arousal,
                    'valence': agent.emotional_state.valence,
                    'anxiety': agent.emotional_state.anxiety,
                    'confidence': agent.emotional_state.confidence,
                    'social_energy': agent.emotional_state.social_energy,
                    'cognitive_load': agent.emotional_state.cognitive_load
                },
                'created_at': agent.created_at.isoformat(),
                'last_interaction': agent.last_interaction.isoformat() if agent.last_interaction else None
            }
            for agent_id, agent in self.agents.items()
        }
        
        with open(os.path.join(data_dir, 'agents.json'), 'w') as f:
            json.dump(agents_data, f, indent=2)
        
        # Export metrics history
        metrics_data = []
        for metrics in self.metrics_history:
            metrics_data.append({
                'round': metrics.round_number,
                'timestamp': metrics.timestamp.isoformat(),
                'overall_polarization': metrics.overall_polarization,
                'overall_consensus': metrics.overall_consensus,
                'avg_certainty': metrics.avg_certainty,
                'avg_valence': metrics.avg_emotional_valence,
                'interaction_count': metrics.interaction_count,
                'topic_metrics': metrics.opinion_metrics,
                'cluster_evolution': {topic: data.get('cluster_summaries', []) 
                                    for topic, data in metrics.opinion_metrics.items()}
            })
        
        with open(os.path.join(data_dir, 'metrics_history.json'), 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        # Export full conversation histories
        conversations_data = []
        for conv in self.conversations:
            conversation_data = {
                'conversation_id': id(conv),
                'participants': conv.participants,
                'participant_names': [self.agents[pid].name for pid in conv.participants],
                'timestamp': conv.timestamp.isoformat(),
                'duration_turns': conv.duration_turns,
                'plan': {
                    'topics': conv.plan.topics,
                    'intents': conv.plan.intents,
                    'expected_duration': conv.plan.expected_duration
                },
                'transcript': conv.transcript,  # Full message history
                'analysis': {
                    'overall_quality': conv.analysis.overall_quality,
                    'resolution_achieved': conv.analysis.resolution_achieved,
                    'topics_discussed': conv.analysis.topics_discussed,
                    'agent1_perspective': {
                        'topics_discussed': conv.analysis.agent1_perspective.topics_discussed,
                        'arguments_made': conv.analysis.agent1_perspective.arguments_made,
                        'arguments_encountered': conv.analysis.agent1_perspective.arguments_encountered,
                        'interaction_quality': conv.analysis.agent1_perspective.interaction_quality,
                        'validation_received': conv.analysis.agent1_perspective.validation_received,
                        'conflict_level': conv.analysis.agent1_perspective.conflict_level
                    },
                    'agent2_perspective': {
                        'topics_discussed': conv.analysis.agent2_perspective.topics_discussed,
                        'arguments_made': conv.analysis.agent2_perspective.arguments_made,
                        'arguments_encountered': conv.analysis.agent2_perspective.arguments_encountered,
                        'interaction_quality': conv.analysis.agent2_perspective.interaction_quality,
                        'validation_received': conv.analysis.agent2_perspective.validation_received,
                        'conflict_level': conv.analysis.agent2_perspective.conflict_level
                    }
                },
                'state_changes': {
                    agent_id: {
                        'opinion_changes': changes.opinion_changes,
                        'emotion_changes': changes.emotion_changes
                    }
                    for agent_id, changes in conv.state_changes.items()
                }
            }
            conversations_data.append(conversation_data)
        
        with open(os.path.join(data_dir, 'conversations.json'), 'w') as f:
            json.dump(conversations_data, f, indent=2)
        
        # Export analysis results
        influencers = self.analyzer.identify_influencers(self.conversations, self.agents)
        echo_chambers = self.analyzer.find_echo_chambers(self.agents, self.conversations)
        cluster_report = self.analyzer.generate_cluster_report(self.metrics_history)
        
        analysis_results = {
            'top_influencers': influencers[:10],
            'echo_chambers': echo_chambers,
            'final_metrics': {
                'polarization': self.metrics_history[-1].overall_polarization,
                'consensus': self.metrics_history[-1].overall_consensus,
                'avg_certainty': self.metrics_history[-1].avg_certainty
            },
            'opinion_clusters': cluster_report
        }
        
        with open(os.path.join(data_dir, 'analysis.json'), 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        print(f"Exported results to {output_dir}/data/")
    
    def _serialize_agents(self) -> List[Dict]:
        """Serialize agents for checkpoint."""
        return [
            {
                'id': agent.id,
                'name': agent.name,
                'personality': agent.personality.__dict__,
                'background': {
                    'age': agent.background.age,
                    'occupation': agent.background.occupation,
                    'education_level': agent.background.education_level,
                    'education_field': agent.background.education_field,
                    'tags': {
                        'socioeconomic': agent.background.socioeconomic_tags,
                        'relationship': agent.background.relationship_tags,
                        'cultural': agent.background.cultural_tags
                    }
                },
                'opinions': {
                    topic: opinion.__dict__ 
                    for topic, opinion in agent.opinions.items()
                },
                'emotional_state': agent.emotional_state.__dict__
            }
            for agent in self.agents.values()
        ]
    
    def _serialize_metrics(self) -> List[Dict]:
        """Serialize metrics history for checkpoint."""
        return [
            {
                'round': m.round_number,
                'polarization': m.overall_polarization,
                'consensus': m.overall_consensus,
                'certainty': m.avg_certainty,
                'valence': m.avg_emotional_valence,
                'interactions': m.interaction_count
            }
            for m in self.metrics_history
        ]
    
    def __del__(self):
        """Cleanup async resources on deletion."""
        if hasattr(self, 'async_orchestrator') and self.async_orchestrator:
            self.async_orchestrator.cleanup()