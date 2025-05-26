"""Pre-interaction planning for agent conversations."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import random
import logging
from ..agents.profile import Agent
from ..agents.personality import PersonalityBehaviorOntology

logger = logging.getLogger('dynavox.interactions')


@dataclass
class InteractionPlan:
    """Plan for an interaction between agents."""
    topics: List[str]
    intents: Dict[str, str]  # agent_id -> intent
    context: Dict = field(default_factory=dict)
    expected_duration: int = 10  # turns
    

class InteractionPlanner:
    """Plans interactions between agents."""
    
    def __init__(self):
        self.ontology = PersonalityBehaviorOntology()
    
    def plan_interaction(self, agent1: Agent, agent2: Agent, 
                        context: Optional[Dict] = None) -> InteractionPlan:
        """Determine topics and intents for the conversation."""
        context = context or {}
        logger.debug(f"Planning interaction between {agent1.name} and {agent2.name}")
        
        # Find overlapping topics of interest
        shared_topics = set(agent1.opinions.keys()) & set(agent2.opinions.keys())
        logger.debug(f"Found {len(shared_topics)} shared topics: {shared_topics}")
        
        if not shared_topics:
            raise ValueError("Agents have no shared topics to discuss")
        
        # Select topics based on importance and emotional charge
        topic_scores = self._score_topics(agent1, agent2, shared_topics)
        selected_topics = self._select_topics(topic_scores, context)
        
        # Determine intents based on personality
        intent1 = self._determine_intent(agent1, agent2, selected_topics)
        intent2 = self._determine_intent(agent2, agent1, selected_topics)
        
        # Estimate conversation duration based on personalities
        expected_duration = self._estimate_duration(agent1, agent2)
        
        return InteractionPlan(
            topics=selected_topics,
            intents={agent1.id: intent1, agent2.id: intent2},
            context=context,
            expected_duration=expected_duration
        )
    
    def _score_topics(self, agent1: Agent, agent2: Agent, 
                     shared_topics: set) -> Dict[str, float]:
        """Score topics based on relevance and potential for engagement."""
        topic_scores = {}
        
        for topic in shared_topics:
            # Base score from importance and emotional charge
            score = (
                agent1.opinions[topic].importance + 
                agent2.opinions[topic].importance +
                agent1.opinions[topic].emotional_charge + 
                agent2.opinions[topic].emotional_charge
            ) / 4
            
            # Bonus for disagreement (creates discussion)
            position_diff = abs(agent1.opinions[topic].position - 
                              agent2.opinions[topic].position)
            if position_diff > 30:  # Significant disagreement
                score *= 1.2
            
            # Penalty if both have very low knowledge
            if (agent1.opinions[topic].knowledge < 30 and 
                agent2.opinions[topic].knowledge < 30):
                score *= 0.7
            
            # Boost if topic matches current emotional states
            if agent1.emotional_state.anxiety > 60 or agent2.emotional_state.anxiety > 60:
                # Anxious agents avoid highly charged topics
                if agent1.opinions[topic].emotional_charge > 70 or \
                   agent2.opinions[topic].emotional_charge > 70:
                    score *= 0.5
            
            topic_scores[topic] = score
        
        return topic_scores
    
    def _select_topics(self, topic_scores: Dict[str, float], 
                      context: Dict) -> List[str]:
        """Select final topics for discussion."""
        # Sort topics by score
        sorted_topics = sorted(topic_scores.items(), 
                             key=lambda x: x[1], reverse=True)
        
        # Number of topics depends on context
        max_topics = context.get('max_topics', 3)
        min_score = context.get('min_topic_score', 20)
        
        selected = []
        for topic, score in sorted_topics:
            if score >= min_score and len(selected) < max_topics:
                selected.append(topic)
        
        # Ensure at least one topic
        if not selected and sorted_topics:
            selected.append(sorted_topics[0][0])
        
        return selected
    
    def _determine_intent(self, agent: Agent, partner: Agent, 
                         topics: List[str]) -> str:
        """Determine agent's conversational intent."""
        # Get personality-based probabilities
        intent_probs = self.ontology.get_conversation_intent_probability(
            agent.personality)
        
        # Modify based on current state
        if agent.emotional_state.anxiety > 70:
            intent_probs['validate'] *= 1.5
            intent_probs['debate'] *= 0.5
        
        if agent.emotional_state.confidence > 70:
            intent_probs['persuade'] *= 1.3
            intent_probs['learn'] *= 0.8
        
        if agent.emotional_state.social_energy < 30:
            intent_probs['bond'] *= 0.5
            intent_probs['explore'] *= 0.7
        
        # Consider topic characteristics
        avg_certainty = sum(agent.opinions[t].certainty for t in topics) / len(topics)
        if avg_certainty > 80:
            intent_probs['persuade'] *= 1.2
            intent_probs['learn'] *= 0.7
        elif avg_certainty < 40:
            intent_probs['learn'] *= 1.3
            intent_probs['explore'] *= 1.2
        
        # Normalize probabilities
        total = sum(intent_probs.values())
        intent_probs = {k: v/total for k, v in intent_probs.items()}
        
        # Select intent (could be random weighted choice in production)
        return max(intent_probs.items(), key=lambda x: x[1])[0]
    
    def _estimate_duration(self, agent1: Agent, agent2: Agent) -> int:
        """Estimate expected conversation duration in turns."""
        base_duration = 10
        
        # Extraverts have longer conversations
        extraversion_avg = (agent1.personality.extraversion + 
                           agent2.personality.extraversion) / 2
        duration_modifier = 1 + (extraversion_avg - 50) / 100
        
        # Low social energy shortens conversations
        social_energy_avg = (agent1.emotional_state.social_energy + 
                            agent2.emotional_state.social_energy) / 2
        if social_energy_avg < 30:
            duration_modifier *= 0.7
        
        # High cognitive load shortens conversations
        cognitive_load_avg = (agent1.emotional_state.cognitive_load + 
                             agent2.emotional_state.cognitive_load) / 2
        if cognitive_load_avg > 70:
            duration_modifier *= 0.8
        
        # Agreeable personalities have smoother, longer conversations
        agreeableness_avg = (agent1.personality.agreeableness + 
                            agent2.personality.agreeableness) / 2
        if agreeableness_avg > 70:
            duration_modifier *= 1.2
        
        return int(base_duration * duration_modifier)