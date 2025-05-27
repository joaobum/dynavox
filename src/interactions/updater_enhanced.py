"""Enhanced post-interaction state update mechanisms with more dynamic opinion evolution."""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple
import numpy as np
import logging
from ..agents.profile import Agent
from ..agents.personality import PersonalityBehaviorOntology
from ..llm.client import LLMClient
from ..llm.prompts import PromptTemplates
from .planner import InteractionPlan
from .updater import StateUpdater, AgentPerspective, ConversationAnalysis, StateChange

logger = logging.getLogger('dynavox.interactions')


class EnhancedStateUpdater(StateUpdater):
    """Enhanced state updater with more realistic opinion dynamics."""
    
    def calculate_state_changes(self, agent: Agent, partner: Agent,
                              analysis: ConversationAnalysis,
                              perspective: AgentPerspective) -> StateChange:
        """Calculate how an agent's state should change with enhanced dynamics."""
        logger.debug(f"Calculating enhanced state changes for {agent.name} after conversation with {partner.name}")
        state_change = StateChange()
        
        # Analyze conversation dynamics
        validation_level = perspective.validation_received
        conflict_level = perspective.conflict_level
        interaction_quality = perspective.interaction_quality
        
        # Determine conversation type
        conversation_type = self._classify_conversation_type(
            validation_level, conflict_level, interaction_quality
        )
        logger.debug(f"  Conversation type: {conversation_type}")
        
        # Opinion changes based on enhanced influence factors
        for topic in perspective.topics_discussed:
            if topic not in agent.opinions:
                continue
            
            logger.debug(f"  Evaluating enhanced influence on {topic}")
            
            # Calculate influence with enhanced dynamics
            influence = self._calculate_enhanced_influence(
                agent, partner, topic, 
                perspective.arguments_encountered.get(topic, []),
                perspective.interaction_quality,
                conversation_type,
                validation_level,
                conflict_level
            )
            
            logger.debug(f"    Position delta: {influence['position_delta']:.2f}, "
                        f"Certainty delta: {influence['certainty_delta']:.2f}, "
                        f"Importance delta: {influence['importance_delta']:.2f}")
            
            # Apply influence with lower threshold (was > 2, now > 0.5)
            if abs(influence['position_delta']) > 0.5 or \
               abs(influence['certainty_delta']) > 1 or \
               abs(influence['importance_delta']) > 1:
                state_change.opinion_changes[topic] = {
                    'position': influence['position_delta'],
                    'certainty': influence['certainty_delta'],
                    'importance': influence['importance_delta']
                }
        
        # Enhanced emotional state changes
        emotion_deltas = self._calculate_enhanced_emotional_impact(
            agent,
            perspective.interaction_quality,
            perspective.validation_received,
            perspective.conflict_level,
            conversation_type
        )
        
        state_change.emotion_changes = emotion_deltas
        
        return state_change
    
    def _classify_conversation_type(self, validation: float, conflict: float, 
                                  quality: float) -> str:
        """Classify the type of conversation for enhanced dynamics."""
        if validation > 0.7 and conflict < 0.3:
            return "echo_chamber"  # Mutual reinforcement
        elif validation < 0.3 and conflict > 0.7:
            return "heated_debate"  # Strong disagreement
        elif validation > 0.5 and conflict > 0.5:
            return "respectful_disagreement"  # Disagree but validate
        elif quality > 0.7:
            return "productive_exchange"  # High quality discussion
        else:
            return "neutral_conversation"  # Standard interaction
    
    def _calculate_enhanced_influence(self, agent: Agent, partner: Agent,
                                    topic: str, arguments: List[str],
                                    interaction_quality: float,
                                    conversation_type: str,
                                    validation_level: float,
                                    conflict_level: float) -> Dict[str, float]:
        """Enhanced influence calculation with conversation dynamics."""
        # Get base influence from parent method
        base_result = super()._calculate_influence(agent, partner, topic, 
                                                 arguments, interaction_quality)
        
        # Extract base values
        base_position_delta = base_result['position_delta']
        base_certainty_delta = base_result['certainty_delta']
        base_importance_delta = base_result['importance_delta']
        
        # Enhanced position changes based on conversation type
        position_multiplier = 1.0
        certainty_modifier = 0
        importance_modifier = 0
        
        if conversation_type == "echo_chamber":
            # Validation strengthens existing views
            if np.sign(agent.opinions[topic].position) == np.sign(partner.opinions[topic].position):
                # Same side - increase extremity
                position_multiplier = 0.3  # Less position change
                current_position = agent.opinions[topic].position
                if abs(current_position) < 80:  # Room to become more extreme
                    position_delta = np.sign(current_position) * 5 * validation_level
                else:
                    position_delta = base_position_delta * position_multiplier
                certainty_modifier = 10 * validation_level  # Much more certain
                importance_modifier = 5 * validation_level  # More important
            else:
                # Opposite sides but validating - cognitive dissonance
                position_multiplier = 1.5  # More position change
                position_delta = base_position_delta * position_multiplier
                certainty_modifier = -5  # Less certain due to dissonance
                importance_modifier = 3  # Still becomes important
        
        elif conversation_type == "heated_debate":
            # Conflict can entrench or shift views dramatically
            if agent.personality.agreeableness < 40:  # Disagreeable - entrench
                position_multiplier = -0.5  # Backlash effect
                position_delta = -base_position_delta * 0.5
                certainty_modifier = 5  # More certain of own view
                importance_modifier = 10 * conflict_level  # Very important now
            else:  # Agreeable - may shift to reduce conflict
                position_multiplier = 2.0  # Larger position change
                position_delta = base_position_delta * position_multiplier
                certainty_modifier = -10 * conflict_level  # Much less certain
                importance_modifier = 8 * conflict_level  # Important due to stress
        
        elif conversation_type == "respectful_disagreement":
            # Most conducive to genuine opinion change
            position_multiplier = 1.8  # Enhanced position change
            position_delta = base_position_delta * position_multiplier
            certainty_modifier = -3  # Slightly less certain (open to change)
            importance_modifier = 7  # More important (engaged topic)
        
        elif conversation_type == "productive_exchange":
            # Quality discussion enhances all changes
            position_multiplier = 1.5
            position_delta = base_position_delta * position_multiplier
            certainty_modifier = 5 if arguments else 2
            importance_modifier = 5
        
        else:  # neutral_conversation
            position_delta = base_position_delta
            certainty_modifier = base_certainty_delta
            importance_modifier = base_importance_delta
        
        # Apply personality modifiers
        if agent.personality.openness > 70:
            position_multiplier *= 1.3  # More open to change
        elif agent.personality.openness < 30:
            position_multiplier *= 0.6  # Resistant to change
        
        # Emotional investment affects importance
        current_emotional_charge = agent.opinions[topic].emotional_charge
        if current_emotional_charge > 70:
            importance_modifier += 5  # Already emotionally invested
        
        # High arousal (from any source) makes topics more important
        if agent.emotional_state.arousal > 70:
            importance_modifier += 3
        
        # Social influence for agreeable personalities
        if agent.personality.agreeableness > 70 and validation_level > 0.6:
            position_delta *= 1.4  # More influenced by validation
        
        # Calculate final deltas
        if conversation_type != "echo_chamber" or \
           np.sign(agent.opinions[topic].position) != np.sign(partner.opinions[topic].position):
            final_position_delta = position_delta
        else:
            final_position_delta = position_delta  # Use calculated delta for echo chamber
        
        final_certainty_delta = base_certainty_delta + certainty_modifier
        final_importance_delta = base_importance_delta + importance_modifier
        
        # Ensure minimum meaningful changes for engaged conversations
        if interaction_quality > 0.6:
            if abs(final_position_delta) < 2:
                final_position_delta = np.sign(final_position_delta) * 2
            if abs(final_importance_delta) < 2:
                final_importance_delta = 2
        
        return {
            'position_delta': final_position_delta,
            'certainty_delta': final_certainty_delta,
            'importance_delta': final_importance_delta,
            'influence_strength': base_result['influence_strength'],
            'conversation_type': conversation_type
        }
    
    def _calculate_enhanced_emotional_impact(self, agent: Agent,
                                           interaction_quality: float,
                                           validation_received: float,
                                           conflict_level: float,
                                           conversation_type: str) -> Dict[str, float]:
        """Enhanced emotional impact calculation."""
        # Get base emotional changes
        base_changes = super()._calculate_emotional_impact(
            agent, interaction_quality, validation_received, conflict_level
        )
        
        # Enhance based on conversation type
        if conversation_type == "echo_chamber":
            base_changes['valence'] += 8  # Feel good from validation
            base_changes['confidence'] += 10  # Very confident
            base_changes['social_energy'] += 5  # Energized by agreement
            
        elif conversation_type == "heated_debate":
            base_changes['valence'] -= 5  # Additional negative impact
            base_changes['anxiety'] += 10  # More anxious
            base_changes['cognitive_load'] += 15  # Mentally exhausting
            if agent.personality.extraversion < 40:
                base_changes['social_energy'] -= 10  # Extra draining for introverts
                
        elif conversation_type == "respectful_disagreement":
            base_changes['cognitive_load'] += 10  # Mentally engaging
            base_changes['confidence'] += 3  # Slight confidence boost
            # Valence depends on personality
            if agent.personality.openness > 60:
                base_changes['valence'] += 3  # Enjoy intellectual challenge
                
        elif conversation_type == "productive_exchange":
            base_changes['valence'] += 5  # Positive experience
            base_changes['confidence'] += 5  # Good interaction
            base_changes['cognitive_load'] += 5  # Engaged but not overwhelmed
        
        return base_changes
    
    def apply_state_changes(self, agent: Agent, changes: StateChange) -> None:
        """Apply state changes with enhanced bounds and side effects."""
        # Apply opinion updates with side effects
        for topic, deltas in changes.opinion_changes.items():
            if topic in agent.opinions:
                opinion = agent.opinions[topic]
                
                # Store original values
                original_position = opinion.position
                original_certainty = opinion.certainty
                original_importance = opinion.importance
                
                # Update with bounds checking
                opinion.position = max(-100, min(100, 
                    opinion.position + deltas.get('position', 0)))
                opinion.certainty = max(0, min(100, 
                    opinion.certainty + deltas.get('certainty', 0)))
                opinion.importance = max(0, min(100, 
                    opinion.importance + deltas.get('importance', 0)))
                
                # Update emotional charge based on changes
                position_change = abs(opinion.position - original_position)
                importance_change = opinion.importance - original_importance
                
                if position_change > 10 or importance_change > 5:
                    # Significant change increases emotional charge
                    opinion.emotional_charge = min(100, 
                        opinion.emotional_charge + position_change * 0.5 + importance_change)
                
                # Knowledge might increase from discussion
                if deltas.get('influence_strength', 0) > 0.5:
                    opinion.knowledge = min(100, opinion.knowledge + 3)
        
        # Apply emotional updates with cascading effects
        super().apply_state_changes(agent, changes)
        
        # Additional emotional cascades
        if agent.emotional_state.anxiety > 80:
            # High anxiety reduces confidence
            agent.emotional_state.confidence = max(0, 
                agent.emotional_state.confidence - 5)
        
        if agent.emotional_state.cognitive_load > 80:
            # High cognitive load reduces social energy
            agent.emotional_state.social_energy = max(0,
                agent.emotional_state.social_energy - 5)