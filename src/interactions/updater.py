"""Post-interaction state update mechanisms."""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple
import numpy as np
import logging
from ..agents.profile import Agent
from ..agents.personality import PersonalityBehaviorOntology
from ..llm.client import LLMClient
from ..llm.prompts import PromptTemplates
from .planner import InteractionPlan

logger = logging.getLogger('dynavox.interactions')


@dataclass
class AgentPerspective:
    """Analysis from one agent's perspective."""
    topics_discussed: List[str]
    arguments_made: Dict[str, List[str]] = field(default_factory=dict)
    arguments_encountered: Dict[str, List[str]] = field(default_factory=dict)
    interaction_quality: float = 0.5
    validation_received: float = 0.0
    conflict_level: float = 0.0


@dataclass
class ConversationAnalysis:
    """Complete analysis of a conversation."""
    topics_discussed: List[str]
    agent1_perspective: AgentPerspective
    agent2_perspective: AgentPerspective
    overall_quality: float = 0.5
    resolution_achieved: bool = False


@dataclass 
class StateChange:
    """Record of state changes for an agent."""
    opinion_changes: Dict[str, Dict[str, float]] = field(default_factory=dict)
    emotion_changes: Dict[str, float] = field(default_factory=dict)
    

class StateUpdater:
    """Updates agent states based on conversation outcomes."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.ontology = PersonalityBehaviorOntology()
        self.prompts = PromptTemplates()
    
    def analyze_conversation(self, transcript: List[Dict], 
                           agent1: Agent, agent2: Agent,
                           plan: 'InteractionPlan') -> ConversationAnalysis:
        """Analyze conversation from both agents' perspectives."""
        # Format transcript for analysis
        transcript_text = self._format_transcript(transcript)
        
        # Use LLM to analyze conversation
        analysis_prompt = self.prompts.ANALYZE_CONVERSATION.format(
            agent1_name=agent1.name,
            agent2_name=agent2.name,
            transcript=transcript_text,
            agent1_personality=self._summarize_personality(agent1.personality),
            agent1_intent=plan.intents[agent1.id],
            agent1_opinions=self._summarize_opinions(agent1.opinions, plan.topics),
            agent2_personality=self._summarize_personality(agent2.personality),
            agent2_intent=plan.intents[agent2.id],
            agent2_opinions=self._summarize_opinions(agent2.opinions, plan.topics)
        )
        
        try:
            analysis_data = self.llm.generate_json(analysis_prompt)
            
            # Parse into structured analysis
            return ConversationAnalysis(
                topics_discussed=analysis_data.get('topics_discussed', plan.topics),
                agent1_perspective=AgentPerspective(
                    topics_discussed=analysis_data.get('topics_discussed', plan.topics),
                    arguments_made=self._extract_arguments(analysis_data, 'agent1_perspective', 'arguments_made'),
                    arguments_encountered=self._extract_arguments(analysis_data, 'agent1_perspective', 'arguments_encountered'),
                    interaction_quality=analysis_data.get('agent1_perspective', {}).get('interaction_quality', 0.5),
                    validation_received=analysis_data.get('agent1_perspective', {}).get('validation_received', 0.0),
                    conflict_level=analysis_data.get('agent1_perspective', {}).get('conflict_level', 0.0)
                ),
                agent2_perspective=AgentPerspective(
                    topics_discussed=analysis_data.get('topics_discussed', plan.topics),
                    arguments_made=self._extract_arguments(analysis_data, 'agent2_perspective', 'arguments_made'),
                    arguments_encountered=self._extract_arguments(analysis_data, 'agent2_perspective', 'arguments_encountered'),
                    interaction_quality=analysis_data.get('agent2_perspective', {}).get('interaction_quality', 0.5),
                    validation_received=analysis_data.get('agent2_perspective', {}).get('validation_received', 0.0),
                    conflict_level=analysis_data.get('agent2_perspective', {}).get('conflict_level', 0.0)
                ),
                overall_quality=(
                    analysis_data.get('agent1_perspective', {}).get('interaction_quality', 0.5) +
                    analysis_data.get('agent2_perspective', {}).get('interaction_quality', 0.5)
                ) / 2
            )
        except Exception as e:
            # Fallback to simple analysis
            print(f"Warning: LLM analysis failed: {e}. Using fallback analysis.")
            return self._fallback_analysis(transcript, agent1, agent2, plan)
    
    def calculate_state_changes(self, agent: Agent, partner: Agent,
                              analysis: ConversationAnalysis,
                              perspective: AgentPerspective) -> StateChange:
        """Calculate how an agent's state should change."""
        logger.debug(f"Calculating state changes for {agent.name} after conversation with {partner.name}")
        state_change = StateChange()
        
        # Opinion changes based on influence factors
        for topic in perspective.topics_discussed:
            if topic not in agent.opinions:
                continue
            
            logger.debug(f"  Evaluating influence on {topic}")
            
            # Calculate influence for this topic
            influence = self._calculate_influence(
                agent, partner, topic, 
                perspective.arguments_encountered.get(topic, []),
                perspective.interaction_quality
            )
            logger.debug(f"    Position delta: {influence['position_delta']:.2f}, "
                        f"Certainty delta: {influence['certainty_delta']:.2f}")
            
            # Apply influence if significant
            if abs(influence['position_delta']) > 2:
                state_change.opinion_changes[topic] = {
                    'position': influence['position_delta'],
                    'certainty': influence['certainty_delta'],
                    'importance': influence['importance_delta']
                }
        
        # Emotional state changes
        emotion_deltas = self._calculate_emotional_impact(
            agent,
            perspective.interaction_quality,
            perspective.validation_received,
            perspective.conflict_level
        )
        
        state_change.emotion_changes = emotion_deltas
        
        return state_change
    
    def apply_state_changes(self, agent: Agent, changes: StateChange) -> None:
        """Apply calculated state changes to agent."""
        # Apply opinion updates
        for topic, deltas in changes.opinion_changes.items():
            if topic in agent.opinions:
                opinion = agent.opinions[topic]
                
                # Update with bounds checking
                opinion.position = max(-100, min(100, 
                    opinion.position + deltas.get('position', 0)))
                opinion.certainty = max(0, min(100, 
                    opinion.certainty + deltas.get('certainty', 0)))
                opinion.importance = max(0, min(100, 
                    opinion.importance + deltas.get('importance', 0)))
        
        # Apply emotional updates
        for field, delta in changes.emotion_changes.items():
            if hasattr(agent.emotional_state, field):
                current = getattr(agent.emotional_state, field)
                if field == 'valence':
                    new_value = max(-50, min(50, current + delta))
                else:
                    new_value = max(0, min(100, current + delta))
                setattr(agent.emotional_state, field, new_value)
    
    def _calculate_influence(self, agent: Agent, partner: Agent,
                           topic: str, arguments: List[str],
                           interaction_quality: float) -> Dict[str, float]:
        """Calculate opinion change based on personality and arguments."""
        # Get personality-based modifiers
        modifiers = self.ontology.INFLUENCE_MODIFIERS
        
        # Assess source credibility
        credibility = self._assess_credibility(agent, partner, topic)
        
        # Personality-moderated receptivity
        openness_mod = modifiers['openness']['novelty_bonus'](agent.personality.openness)
        certainty_barrier = modifiers['openness']['certainty_flexibility'](agent.personality.openness)
        evidence_need = modifiers['conscientiousness']['evidence_requirement'](agent.personality.conscientiousness)
        
        # Evaluate argument quality
        argument_quality = self._evaluate_argument_quality(
            arguments, agent, topic) if arguments else 0.3
        
        # Social factors
        social_influence = 0.0
        if agent.personality.agreeableness > 60 and interaction_quality > 0.7:
            social_influence = modifiers['agreeableness']['empathy_driven_change'](
                agent.personality.agreeableness)
        
        # Calculate base influence
        base_influence = (credibility * 0.4 + 
                         argument_quality * 0.4 + 
                         social_influence * 0.2) * (1 + openness_mod)
        
        # Apply certainty resistance
        certainty_resistance = (agent.opinions[topic].certainty / 100) * certainty_barrier
        effective_influence = base_influence * (1 - certainty_resistance)
        
        # Apply interaction quality modifier
        effective_influence *= interaction_quality
        
        # Calculate position change
        position_diff = partner.opinions[topic].position - agent.opinions[topic].position
        position_delta = position_diff * effective_influence * 0.15  # Max 15% change
        
        # Certainty changes
        if argument_quality > evidence_need:
            certainty_delta = 5 * interaction_quality
        else:
            certainty_delta = -3 if effective_influence < 0.3 else 0
        
        # Importance changes
        importance_delta = 0
        if abs(position_delta) > 5 or interaction_quality > 0.8:
            importance_delta = 3
        
        return {
            'position_delta': position_delta,
            'certainty_delta': certainty_delta,
            'importance_delta': importance_delta,
            'influence_strength': effective_influence
        }
    
    def _calculate_emotional_impact(self, agent: Agent,
                                  interaction_quality: float,
                                  validation_received: float,
                                  conflict_level: float) -> Dict[str, float]:
        """Calculate emotional state changes from interaction."""
        changes = {}
        
        # Valence changes
        valence_delta = 0
        if interaction_quality > 0.7:
            valence_delta += 5
        if validation_received > 0.5:
            valence_delta += 5
        if conflict_level > 0.7:
            valence_delta -= 10
            
        # Personality moderation
        if agent.personality.emotionality > 70:
            valence_delta *= 1.5  # More reactive
        
        changes['valence'] = valence_delta
        
        # Anxiety changes
        anxiety_delta = 0
        if conflict_level > 0.5:
            anxiety_delta += 10
        if interaction_quality < 0.3:
            anxiety_delta += 5
        if validation_received > 0.7:
            anxiety_delta -= 5
            
        # Low stress tolerance increases anxiety response
        if agent.emotional_baseline.stress_tolerance < 40:
            anxiety_delta *= 1.3
            
        changes['anxiety'] = anxiety_delta
        
        # Confidence changes
        confidence_delta = 0
        if validation_received > 0.6:
            confidence_delta += 8
        if interaction_quality > 0.7:
            confidence_delta += 3
        if conflict_level > 0.7 and agent.personality.agreeableness > 60:
            confidence_delta -= 5
            
        changes['confidence'] = confidence_delta
        
        # Social energy changes
        social_energy_delta = 0
        if agent.personality.extraversion > 60:
            if interaction_quality > 0.5:
                social_energy_delta += 5  # Energized
        else:
            social_energy_delta -= 5  # Drained
            
        if conflict_level > 0.7:
            social_energy_delta -= 5
            
        changes['social_energy'] = social_energy_delta
        
        # Cognitive load changes
        cognitive_load_delta = 0
        if conflict_level > 0.5:
            cognitive_load_delta += 10
        if len(agent.opinions) > 5:  # Many topics discussed
            cognitive_load_delta += 5
            
        changes['cognitive_load'] = cognitive_load_delta
        
        return changes
    
    def _assess_credibility(self, agent: Agent, partner: Agent, 
                          topic: str) -> float:
        """Assess partner's credibility from agent's perspective."""
        credibility = 0.5  # Base credibility
        
        # Education level comparison
        edu_levels = ['high_school', 'bachelors', 'masters', 'phd']
        agent_edu_idx = edu_levels.index(agent.background.education_level)
        partner_edu_idx = edu_levels.index(partner.background.education_level)
        
        if partner_edu_idx > agent_edu_idx:
            credibility += 0.1
        
        # Knowledge comparison
        if topic in partner.opinions and topic in agent.opinions:
            knowledge_diff = partner.opinions[topic].knowledge - agent.opinions[topic].knowledge
            if knowledge_diff > 20:
                credibility += 0.2
            elif knowledge_diff < -20:
                credibility -= 0.1
        
        # Personality trust factors
        if agent.personality.agreeableness > 70:
            credibility += 0.1  # Trusting nature
        if agent.personality.honesty_humility > 70:
            # Values honesty, assesses it in others
            if partner.personality.honesty_humility > 60:
                credibility += 0.1
        
        # Professional relevance
        if self._is_topic_professionally_relevant(partner, topic):
            credibility += 0.15
        
        return max(0.1, min(0.9, credibility))
    
    def _evaluate_argument_quality(self, arguments: List[str], 
                                 agent: Agent, topic: str) -> float:
        """Evaluate argument quality from agent's perspective."""
        if not arguments:
            return 0.3
        
        # Use LLM to evaluate arguments
        prompt = self.prompts.EVALUATE_ARGUMENT_QUALITY.format(
            speaker_name="the other person",
            education=f"{agent.background.education_level} in {agent.background.education_field}",
            openness=agent.personality.openness,
            conscientiousness=agent.personality.conscientiousness,
            knowledge=agent.opinions[topic].knowledge if topic in agent.opinions else 50,
            arguments="\n".join(f"- {arg}" for arg in arguments[:3])  # Limit to top 3
        )
        
        try:
            quality_str = self.llm.generate(prompt)
            # Extract numeric value
            quality = float(quality_str.strip())
            return max(0.0, min(1.0, quality))
        except:
            # Fallback: simple heuristic
            base_quality = 0.5
            
            # High conscientiousness demands better arguments
            if agent.personality.conscientiousness > 70:
                base_quality -= 0.1
            
            # High openness more receptive
            if agent.personality.openness > 70:
                base_quality += 0.1
                
            return base_quality
    
    def _format_transcript(self, transcript: List[Dict]) -> str:
        """Format transcript for analysis."""
        lines = []
        for turn in transcript:
            speaker = turn.get('speaker', 'Unknown')
            content = turn.get('content', '')
            lines.append(f"{speaker}: {content}")
        return "\n".join(lines)
    
    def _summarize_personality(self, personality) -> str:
        """Create brief personality summary."""
        traits = []
        if personality.extraversion > 70:
            traits.append("highly extraverted")
        elif personality.extraversion < 30:
            traits.append("introverted")
            
        if personality.agreeableness > 70:
            traits.append("very agreeable")
        elif personality.agreeableness < 30:
            traits.append("disagreeable")
            
        if personality.openness > 70:
            traits.append("very open")
        elif personality.openness < 30:
            traits.append("traditional")
            
        return ", ".join(traits) if traits else "balanced personality"
    
    def _summarize_opinions(self, opinions: Dict, topics: List[str]) -> str:
        """Summarize relevant opinions."""
        summaries = []
        for topic in topics:
            if topic in opinions:
                op = opinions[topic]
                position_desc = "strongly supports" if op.position > 50 else \
                               "strongly opposes" if op.position < -50 else \
                               "is neutral on"
                summaries.append(f"{position_desc} {topic}")
        return "; ".join(summaries)
    
    def _extract_arguments(self, data: Dict, perspective: str, 
                          argument_type: str) -> Dict[str, List[str]]:
        """Extract arguments from analysis data."""
        args = data.get(perspective, {}).get(argument_type, [])
        
        # Group by topic if possible
        topic_args = {}
        for arg in args:
            # Simple heuristic: assign to first mentioned topic
            assigned = False
            for topic in data.get('topics_discussed', []):
                if topic.replace('_', ' ') in arg.lower():
                    if topic not in topic_args:
                        topic_args[topic] = []
                    topic_args[topic].append(arg)
                    assigned = True
                    break
            
            if not assigned and data.get('topics_discussed'):
                # Assign to first topic as default
                topic = data['topics_discussed'][0]
                if topic not in topic_args:
                    topic_args[topic] = []
                topic_args[topic].append(arg)
        
        return topic_args
    
    def _is_topic_professionally_relevant(self, agent: Agent, topic: str) -> bool:
        """Check if topic is relevant to agent's profession."""
        occupation_lower = agent.background.occupation.lower()
        
        relevance_map = {
            'climate_change': [
                # Environmental professions
                'environmental', 'scientist', 'policy', 'energy', 'climate',
                'meteorologist', 'geologist', 'ecologist', 'conservation',
                'renewable', 'sustainability', 'green', 'solar', 'wind',
                # Agricultural professions
                'farmer', 'agriculture', 'rancher', 'forestry',
                # Engineering related
                'engineer', 'architect', 'urban planner',
                # Education
                'professor', 'teacher', 'researcher'
            ],
            'universal_healthcare': [
                # Medical professions
                'doctor', 'nurse', 'healthcare', 'medical', 'physician',
                'surgeon', 'therapist', 'pharmacist', 'dentist', 'veterinarian',
                'psychiatrist', 'psychologist', 'counselor', 'social worker',
                # Healthcare support
                'technician', 'aide', 'assistant', 'paramedic', 'emt',
                # Administrative
                'insurance', 'billing', 'administrator', 'health',
                # Policy
                'policy', 'public health', 'epidemiologist'
            ],
            'remote_work': [
                # Tech professions
                'software', 'tech', 'it', 'developer', 'programmer',
                'engineer', 'designer', 'analyst', 'data', 'cyber',
                'web', 'mobile', 'cloud', 'devops', 'qa', 'tester',
                # Creative professions
                'writer', 'editor', 'content', 'marketing', 'graphic',
                'video', 'photographer', 'artist', 'musician',
                # Business roles
                'consultant', 'project manager', 'product', 'sales',
                'customer', 'support', 'account', 'recruiter',
                # Education
                'teacher', 'tutor', 'instructor', 'professor'
            ],
            'ai_regulation': [
                # AI/ML professions
                'ai', 'artificial intelligence', 'machine learning', 'ml',
                'data scientist', 'researcher', 'engineer',
                # Tech leadership
                'cto', 'cio', 'tech lead', 'architect',
                # Legal professions
                'lawyer', 'attorney', 'legal', 'compliance', 'regulator',
                'policy', 'ethics', 'philosopher',
                # Business strategy
                'consultant', 'strategist', 'analyst',
                # Affected professions
                'writer', 'artist', 'translator', 'driver', 'accountant'
            ],
            'wealth_inequality': [
                # Economic professions
                'economist', 'finance', 'banker', 'investor', 'trader',
                'analyst', 'accountant', 'advisor', 'wealth',
                # Policy and social
                'policy', 'social', 'nonprofit', 'charity', 'advocate',
                'community', 'organizer', 'activist',
                # Academic
                'professor', 'researcher', 'sociologist', 'political',
                # Service industry
                'retail', 'food', 'service', 'hospitality', 'minimum wage',
                # Labor
                'union', 'labor', 'worker', 'employee'
            ],
            'immigration': [
                # Legal professions
                'immigration', 'lawyer', 'attorney', 'legal', 'paralegal',
                # Government
                'border', 'customs', 'government', 'policy', 'diplomat',
                # Social services
                'social worker', 'translator', 'interpreter', 'advocate',
                # Business
                'hr', 'human resources', 'recruiter', 'employer',
                # Affected industries
                'agriculture', 'construction', 'hospitality', 'restaurant'
            ],
            'education_reform': [
                # Education professions
                'teacher', 'professor', 'educator', 'principal', 'dean',
                'administrator', 'counselor', 'tutor', 'instructor',
                # Support roles
                'aide', 'librarian', 'coach', 'mentor',
                # Policy
                'policy', 'superintendent', 'board',
                # Parents
                'parent', 'mother', 'father'
            ],
            'gun_control': [
                # Law enforcement
                'police', 'officer', 'sheriff', 'detective', 'security',
                'military', 'veteran', 'soldier',
                # Legal
                'lawyer', 'attorney', 'judge', 'prosecutor',
                # Medical
                'emergency', 'trauma', 'surgeon', 'emt', 'paramedic',
                # Education
                'teacher', 'principal', 'school',
                # Retail
                'gun', 'sporting goods', 'firearms'
            ]
        }
        
        relevant_terms = relevance_map.get(topic, [])
        return any(term in occupation_lower for term in relevant_terms)
    
    def _fallback_analysis(self, transcript: List[Dict], 
                          agent1: Agent, agent2: Agent,
                          plan: 'InteractionPlan') -> ConversationAnalysis:
        """Simple fallback analysis when LLM fails."""
        # Count turns and estimate quality
        agent1_turns = sum(1 for t in transcript if t.get('speaker_id') == agent1.id)
        agent2_turns = sum(1 for t in transcript if t.get('speaker_id') == agent2.id)
        
        # Balanced conversation is higher quality
        balance = 1 - abs(agent1_turns - agent2_turns) / max(agent1_turns + agent2_turns, 1)
        
        return ConversationAnalysis(
            topics_discussed=plan.topics,
            agent1_perspective=AgentPerspective(
                topics_discussed=plan.topics,
                interaction_quality=balance,
                validation_received=0.5,
                conflict_level=0.3
            ),
            agent2_perspective=AgentPerspective(
                topics_discussed=plan.topics,
                interaction_quality=balance,
                validation_received=0.5,
                conflict_level=0.3
            ),
            overall_quality=balance
        )