"""Main conversation orchestration between agents."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
from ..agents.profile import Agent
from ..agents.personality import PersonalityBehaviorOntology
from ..llm.client import LLMClient
from ..llm.prompts import PromptTemplates
from .planner import InteractionPlanner, InteractionPlan
from .updater import StateUpdater, ConversationAnalysis, StateChange

logger = logging.getLogger('dynavox.interactions')


@dataclass
class Conversation:
    """Complete record of a conversation between agents."""
    participants: List[str]  # agent IDs
    plan: InteractionPlan
    transcript: List[Dict[str, Any]]
    analysis: ConversationAnalysis
    state_changes: Dict[str, StateChange]
    timestamp: datetime = field(default_factory=datetime.now)
    duration_turns: int = 0


class ConversationOrchestrator:
    """Manages agent conversations from planning through execution to updates."""
    
    def __init__(self, llm_client: LLMClient, max_turns: int = 20):
        self.llm = llm_client
        self.ontology = PersonalityBehaviorOntology()
        logger.debug(f"Initialized ConversationOrchestrator with max_turns={max_turns}")
        self.prompts = PromptTemplates()
        self.planner = InteractionPlanner()
        self.updater = StateUpdater(llm_client)
        self.max_turns = max_turns
    
    def conduct_conversation(self, agent1: Agent, agent2: Agent, 
                           context: Optional[Dict] = None) -> Conversation:
        """Execute a complete conversation between two agents."""
        logger.debug(f"Starting conversation between {agent1.name} and {agent2.name}")
        
        # Plan the interaction
        logger.debug("Planning interaction")
        plan = self.planner.plan_interaction(agent1, agent2, context)
        
        # Execute the conversation
        logger.debug("Executing conversation")
        transcript = self._execute_conversation(agent1, agent2, plan)
        logger.debug(f"Conversation completed with {len(transcript)} turns")
        
        # Analyze results
        logger.debug("Analyzing conversation")
        analysis = self.updater.analyze_conversation(transcript, agent1, agent2, plan)
        
        # Calculate state changes
        logger.debug("Calculating state changes")
        state_changes = self._calculate_state_changes(agent1, agent2, analysis)
        
        # Apply updates to agents
        logger.debug("Applying updates to agents")
        self._apply_updates(agent1, agent2, state_changes)
        
        # Create conversation object
        conversation = Conversation(
            participants=[agent1.id, agent2.id],
            plan=plan,
            transcript=transcript,
            analysis=analysis,
            state_changes=state_changes,
            duration_turns=len(transcript)
        )
        
        # Print conversation summary and changes
        self._print_conversation_summary(agent1, agent2, conversation)
        
        return conversation
    
    def _execute_conversation(self, agent1: Agent, agent2: Agent,
                            plan: InteractionPlan) -> List[Dict]:
        """Run the actual conversation using LLM."""
        # Create initial prompts for each agent
        prompt1 = self._create_conversation_prompt(agent1, plan.intents[agent1.id], plan)
        prompt2 = self._create_conversation_prompt(agent2, plan.intents[agent2.id], plan)
        
        transcript = []
        conversation_context = []
        
        # Determine who speaks first based on personality
        if agent1.personality.extraversion > agent2.personality.extraversion:
            current_speaker = 1
        elif agent2.personality.extraversion > agent1.personality.extraversion:
            current_speaker = 2
        else:
            # Equal extraversion, more anxious speaks first (seeking validation)
            current_speaker = 1 if agent1.emotional_state.anxiety > agent2.emotional_state.anxiety else 2
        
        # Conversation loop
        for turn_num in range(min(self.max_turns, plan.expected_duration * 2)):
            if current_speaker == 1:
                # Agent 1 speaks
                message = self._generate_message(
                    agent1, prompt1, conversation_context, plan
                )
                
                transcript.append({
                    'speaker': agent1.name,
                    'speaker_id': agent1.id,
                    'content': message,
                    'turn': turn_num
                })
                
                conversation_context.append({
                    'speaker': agent1.name, 
                    'content': message
                })
                
                # Check for natural conclusion
                if self._is_conversation_complete(message, transcript, plan):
                    break
                
                current_speaker = 2
                
            else:
                # Agent 2 speaks
                message = self._generate_message(
                    agent2, prompt2, conversation_context, plan
                )
                
                transcript.append({
                    'speaker': agent2.name,
                    'speaker_id': agent2.id,
                    'content': message,
                    'turn': turn_num
                })
                
                conversation_context.append({
                    'speaker': agent2.name, 
                    'content': message
                })
                
                if self._is_conversation_complete(message, transcript, plan):
                    break
                
                current_speaker = 1
        
        return transcript
    
    def _create_conversation_prompt(self, agent: Agent, intent: str, 
                                  plan: InteractionPlan) -> str:
        """Create the LLM prompt for an agent in conversation."""
        # Get behavioral guidance from ontology
        behavior_implications = self.ontology.describe_personality_implications(
            agent.personality)
        behavior_guide = self._generate_behavior_guide(agent, behavior_implications)
        
        # Format personality description
        personality_desc = self.prompts.format_personality_description(
            agent.personality, behavior_implications)
        
        # Format opinions for topics
        opinions_desc = self._describe_opinions(agent.opinions, plan.topics)
        
        # Describe intent
        intent_desc = self._describe_intent(intent, agent)
        
        prompt = self.prompts.CONVERSATION_PROMPT.format(
            name=agent.name,
            age=agent.background.age,
            occupation=agent.background.occupation,
            biography=agent.biography,
            personality_description=personality_desc,
            arousal=agent.emotional_state.arousal,
            valence=agent.emotional_state.valence,
            anxiety=agent.emotional_state.anxiety,
            confidence=agent.emotional_state.confidence,
            opinions_description=opinions_desc,
            intent=intent,
            intent_description=intent_desc,
            behavior_guide=behavior_guide,
            conversation_style=agent.conversation_style
        )
        
        return prompt
    
    def _generate_message(self, agent: Agent, base_prompt: str,
                         context: List[Dict], plan: InteractionPlan) -> str:
        """Generate a single message from an agent."""
        # Build conversation history
        history = "\n".join([
            f"{msg['speaker']}: {msg['content']}" 
            for msg in context[-10:]  # Last 10 messages for context
        ])
        
        # Add current state modifiers
        state_notes = []
        if agent.emotional_state.anxiety > 70:
            state_notes.append("You're feeling quite anxious right now.")
        if agent.emotional_state.cognitive_load > 70:
            state_notes.append("You're feeling mentally exhausted.")
        if agent.emotional_state.social_energy < 30:
            state_notes.append("You're running low on social energy.")
        
        state_context = " ".join(state_notes) if state_notes else ""
        
        # Build complete prompt
        full_prompt = f"""{base_prompt}

Conversation so far:
{history if history else "You're starting the conversation."}

{state_context}

Your turn to speak as {agent.name}:"""
        
        # Generate response
        try:
            response = self.llm.generate(full_prompt, temperature=0.8, max_tokens=200)
            
            # Clean up response
            response = response.strip()
            
            # Ensure response isn't too long for natural conversation
            sentences = response.split('. ')
            if len(sentences) > 5:
                # Limit based on personality
                max_sentences = 3 if agent.personality.extraversion < 40 else 5
                response = '. '.join(sentences[:max_sentences]) + '.'
            
            return response
            
        except Exception as e:
            # Fallback response
            print(f"Warning: Failed to generate message: {e}")
            return self._generate_fallback_message(agent, plan)
    
    def _is_conversation_complete(self, last_message: str, 
                                transcript: List[Dict],
                                plan: InteractionPlan) -> bool:
        """Determine if conversation should end naturally."""
        # Check for explicit endings
        ending_phrases = [
            "goodbye", "bye", "talk later", "nice talking",
            "gotta go", "need to go", "see you", "take care"
        ]
        
        if any(phrase in last_message.lower() for phrase in ending_phrases):
            return True
        
        # Check if we've covered the planned topics
        if len(transcript) >= plan.expected_duration:
            # But allow some flexibility
            if len(transcript) >= plan.expected_duration * 1.5:
                return True
        
        # Check for natural exhaustion
        if len(transcript) > 5:
            # Look for repetition or short responses
            recent_messages = [t['content'] for t in transcript[-4:]]
            avg_length = sum(len(m.split()) for m in recent_messages) / 4
            
            if avg_length < 10:  # Very short responses
                return True
        
        return False
    
    def _calculate_state_changes(self, agent1: Agent, agent2: Agent,
                               analysis: ConversationAnalysis) -> Dict[str, StateChange]:
        """Calculate state changes for both agents."""
        return {
            agent1.id: self.updater.calculate_state_changes(
                agent1, agent2, analysis, analysis.agent1_perspective
            ),
            agent2.id: self.updater.calculate_state_changes(
                agent2, agent1, analysis, analysis.agent2_perspective
            )
        }
    
    def _apply_updates(self, agent1: Agent, agent2: Agent,
                      state_changes: Dict[str, StateChange]) -> None:
        """Apply calculated changes to agents."""
        self.updater.apply_state_changes(agent1, state_changes[agent1.id])
        self.updater.apply_state_changes(agent2, state_changes[agent2.id])
        
        # Update last interaction time
        agent1.last_interaction = datetime.now()
        agent2.last_interaction = datetime.now()
    
    def _generate_behavior_guide(self, agent: Agent, 
                               implications: Dict) -> str:
        """Generate specific behavioral guidance for the agent."""
        guides = []
        
        # Keep guidance SHORT and NATURAL
        # Extraversion guidance
        if agent.personality.extraversion > 70:
            guides.append("You talk easily and share thoughts freely")
        elif agent.personality.extraversion < 30:
            guides.append("You're more of a listener than a talker")
        
        # Agreeableness guidance  
        if agent.personality.agreeableness > 70:
            guides.append("You naturally seek common ground")
        elif agent.personality.agreeableness < 30:
            guides.append("You say what you think, even if others disagree")
        
        # Openness guidance
        if agent.personality.openness > 70:
            guides.append("You enjoy exploring new angles and possibilities")
        elif agent.personality.openness < 30:
            guides.append("You prefer practical, proven approaches")
        
        # Conscientiousness guidance
        if agent.personality.conscientiousness > 70:
            guides.append("You think before you speak and like being accurate")
        elif agent.personality.conscientiousness < 30:
            guides.append("You speak off the cuff without overthinking")
        
        # Emotionality guidance
        if agent.personality.emotionality > 70:
            guides.append("You're emotionally expressive and empathetic")
        elif agent.personality.emotionality < 30:
            guides.append("You stay calm and logical in discussions")
        
        # Current state guidance
        if agent.emotional_state.anxiety > 60:
            guides.append("You're feeling a bit anxious right now")
        
        if agent.emotional_state.confidence > 70:
            guides.append("You're feeling quite confident")
        
        return ". ".join(guides) + "."
    
    def _describe_opinions(self, opinions: Dict[str, Any], topics: List[str]) -> str:
        """Format opinions for the prompt."""
        descriptions = []
        
        for topic in topics:
            if topic not in opinions:
                continue
                
            op = opinions[topic]
            
            # Position description
            if op.position > 50:
                position_desc = "strongly support"
            elif op.position > 20:
                position_desc = "moderately support"
            elif op.position > -20:
                position_desc = "are neutral about"
            elif op.position > -50:
                position_desc = "moderately oppose"
            else:
                position_desc = "strongly oppose"
            
            # Certainty qualifier
            if op.certainty < 40:
                certainty_desc = "but you're not very sure"
            elif op.certainty > 70:
                certainty_desc = "and you're quite certain"
            else:
                certainty_desc = ""
            
            # Importance note
            if op.importance > 70:
                importance_desc = "This is very important to you"
            elif op.importance < 30:
                importance_desc = "This isn't a priority for you"
            else:
                importance_desc = ""
            
            # Build complete description
            desc_parts = [f"You {position_desc} {topic.replace('_', ' ')}"]
            if certainty_desc:
                desc_parts[0] += f" {certainty_desc}"
            if importance_desc:
                desc_parts.append(importance_desc)
            
            descriptions.append(". ".join(desc_parts) + ".")
        
        return " ".join(descriptions)
    
    def _describe_intent(self, intent: str, agent: Agent) -> str:
        """Provide detailed description of conversation intent."""
        intent_descriptions = {
            'learn': "You're curious about their perspective",
            'persuade': "You'd like to share your viewpoint and see if they agree",
            'validate': "You're looking for someone who understands",
            'bond': "You want to connect and find common ground",
            'debate': "You enjoy a good discussion with different viewpoints",
            'explore': "You're thinking out loud and open to new ideas"
        }
        
        base_desc = intent_descriptions.get(intent, "chat naturally")
        
        # Add simple personality-based modifier
        if intent == 'persuade' and agent.personality.agreeableness > 70:
            base_desc += " (but gently, in your style)"
        elif intent == 'debate' and agent.personality.emotionality > 70:
            base_desc += " (while staying emotionally aware)"
        
        return base_desc
    
    def _generate_fallback_message(self, agent: Agent, plan: InteractionPlan) -> str:
        """Generate a simple fallback message when LLM fails."""
        # Simple templates based on personality
        if agent.personality.extraversion > 60:
            templates = [
                "That's an interesting point! I've been thinking about this a lot lately.",
                "I see what you mean. From my perspective,",
                "Yes! And another thing to consider is"
            ]
        else:
            templates = [
                "Hmm, I need to think about that.",
                "That's worth considering.",
                "I see your point."
            ]
        
        import random
        return random.choice(templates)
    
    def _print_conversation_summary(self, agent1: Agent, agent2: Agent, 
                                  conversation: Conversation) -> None:
        """Print conversation summary and state changes."""
        logger.info(f"💬 === Conversation Summary: {agent1.name} & {agent2.name} ===")
        
        # Generate brief conversation summary (50 words)
        if len(conversation.transcript) > 0:
            # Extract key points from conversation
            topics_discussed = conversation.plan.topics
            quality = conversation.analysis.overall_quality
            
            # Create summary based on conversation content
            summary_parts = []
            summary_parts.append(f"Discussed {', '.join(topics_discussed)}")
            
            if quality > 0.7:
                summary_parts.append("with good rapport 😊")
            elif quality < 0.3:
                summary_parts.append("with some tension 😟")
            
            if conversation.analysis.resolution_achieved:
                summary_parts.append("and reached mutual understanding 🤝")
            
            summary = ". ".join(summary_parts) + f" ({conversation.duration_turns} turns)"
            logger.info(f"  📝 Summary: {summary}")
        
        # Print opinion changes for each agent
        logger.info(f"  🔄 Opinion Changes:")
        for agent, agent_name in [(agent1, agent1.name), (agent2, agent2.name)]:
            changes = conversation.state_changes.get(agent.id, None)
            if changes and changes.opinion_changes:
                logger.info(f"    {agent_name}:")
                for topic, deltas in changes.opinion_changes.items():
                    position_change = deltas.get('position', 0)
                    certainty_change = deltas.get('certainty', 0)
                    importance_change = deltas.get('importance', 0)
                    
                    # Only print if there were meaningful changes
                    if abs(position_change) > 0.1 or abs(certainty_change) > 0.1:
                        topic_display = topic.replace('_', ' ').title()
                        logger.info(f"      {topic_display}:")
                        if abs(position_change) > 0.1:
                            direction = "➡️" if position_change > 0 else "⬅️"
                            logger.info(f"        {direction} Position: {position_change:+.1f}")
                        if abs(certainty_change) > 0.1:
                            certainty_icon = "🔼" if certainty_change > 0 else "🔽"
                            logger.info(f"        {certainty_icon} Certainty: {certainty_change:+.1f}")
                        if abs(importance_change) > 0.1:
                            importance_icon = "⭐" if importance_change > 0 else "💫"
                            logger.info(f"        {importance_icon} Importance: {importance_change:+.1f}")
            else:
                logger.info(f"    {agent_name}: No significant opinion changes 🔒")
        
        # Print emotional changes for each agent
        logger.info(f"  😊 Emotional Changes:")
        for agent, agent_name in [(agent1, agent1.name), (agent2, agent2.name)]:
            changes = conversation.state_changes.get(agent.id, None)
            if changes and changes.emotion_changes:
                logger.info(f"    {agent_name}:")
                for emotion, delta in changes.emotion_changes.items():
                    if abs(delta) > 0.1:  # Only show meaningful changes
                        emotion_display = emotion.replace('_', ' ').title()
                        # Add emoji based on emotion type and direction
                        if emotion == 'valence':
                            emoji = "😊" if delta > 0 else "😔"
                        elif emotion == 'arousal':
                            emoji = "⚡" if delta > 0 else "😴"
                        elif emotion == 'anxiety':
                            emoji = "😰" if delta > 0 else "😌"
                        elif emotion == 'confidence':
                            emoji = "💪" if delta > 0 else "😟"
                        elif emotion == 'social_energy':
                            emoji = "🔋" if delta > 0 else "🪫"
                        else:
                            emoji = "🧠" if delta > 0 else "🤯"
                        logger.info(f"      {emoji} {emotion_display}: {delta:+.1f}")
            else:
                logger.info(f"    {agent_name}: No significant emotional changes 😐")