"""Async conversation orchestration for parallel interactions."""
import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from ..agents.profile import Agent
from ..llm.async_client import AsyncLLMWrapper
from .orchestrator import ConversationOrchestrator, Conversation
from .planner import InteractionPlanner, InteractionPlan

logger = logging.getLogger(__name__)


class AsyncConversationOrchestrator:
    """Manages agent conversations asynchronously for parallel execution."""
    
    def __init__(self, llm_client, max_turns: int = 20):
        """Initialize async orchestrator.
        
        Args:
            llm_client: Either an AsyncLLMClient or regular LLMClient (will be wrapped)
            max_turns: Maximum turns per conversation
        """
        # Check if client is already async
        if hasattr(llm_client, 'generate_batch'):
            self.async_llm = llm_client
            self.owns_wrapper = False
        else:
            # Wrap synchronous client
            self.async_llm = AsyncLLMWrapper(llm_client)
            self.owns_wrapper = True
        
        # Reuse components from sync orchestrator
        self.sync_orchestrator = ConversationOrchestrator(llm_client, max_turns, use_enhanced_updater=True)
        self.planner = InteractionPlanner()
        # Use the same updater as sync orchestrator
        self.updater = self.sync_orchestrator.updater
        self.max_turns = max_turns
    
    async def conduct_conversations_parallel(self, 
                                           interaction_pairs: List[Tuple[Agent, Agent]],
                                           context: Optional[Dict] = None) -> List[Conversation]:
        """Execute multiple conversations in parallel.
        
        Args:
            interaction_pairs: List of (agent1, agent2) tuples to converse
            context: Optional context for all conversations
            
        Returns:
            List of completed conversations
        """
        logger.info(f"🚀 Starting {len(interaction_pairs)} conversations in parallel...")
        
        # Create conversation status tracking
        conversation_status = {i: "Starting" for i in range(len(interaction_pairs))}
        
        # Create tasks for each conversation with tracking
        tasks = []
        for i, (agent1, agent2) in enumerate(interaction_pairs):
            task = self._conduct_single_conversation_with_tracking(
                i, agent1, agent2, conversation_status, context)
            tasks.append(task)
        
        # Print initial status
        logger.info("  ⚡ All conversations starting simultaneously:")
        for i, (agent1, agent2) in enumerate(interaction_pairs):
            logger.info(f"    [{i+1}] {agent1.name} ↔ {agent2.name}: Starting...")
        
        # Execute all conversations in parallel
        conversations = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Print completion status
        logger.info("✅ All conversations completed in parallel!")
        
        # Filter out any failed conversations
        valid_conversations = []
        for i, conv in enumerate(conversations):
            if isinstance(conv, Exception):
                agent1, agent2 = interaction_pairs[i]
                logger.error(f"  ❌ Conversation failed between {agent1.name} and {agent2.name}: {conv}")
            else:
                valid_conversations.append(conv)
        
        return valid_conversations
    
    async def _conduct_single_conversation_with_tracking(self, conv_id: int, 
                                                        agent1: Agent, agent2: Agent,
                                                        status_dict: Dict[int, str],
                                                        context: Optional[Dict] = None) -> Conversation:
        """Execute a single conversation with status tracking."""
        status_dict[conv_id] = "Planning"
        
        # Plan the interaction (synchronous, fast)
        plan = self.planner.plan_interaction(agent1, agent2, context)
        
        status_dict[conv_id] = "Conversing"
        
        # Execute the conversation asynchronously
        transcript = await self._execute_conversation_async(agent1, agent2, plan)
        
        status_dict[conv_id] = "Analyzing"
        
        # Analyze results (can be async in future)
        analysis = self.updater.analyze_conversation(transcript, agent1, agent2, plan)
        
        # Calculate state changes
        state_changes = self.sync_orchestrator._calculate_state_changes(agent1, agent2, analysis)
        
        # Apply updates to agents
        self.sync_orchestrator._apply_updates(agent1, agent2, state_changes)
        
        # Create conversation object
        conversation = Conversation(
            participants=[agent1.id, agent2.id],
            plan=plan,
            transcript=transcript,
            analysis=analysis,
            state_changes=state_changes,
            duration_turns=len(transcript)
        )
        
        status_dict[conv_id] = "Complete"
        
        # Don't print summary here - it would break parallel display
        # Summary will be printed after all conversations complete
        
        return conversation
    
    async def _conduct_single_conversation(self, agent1: Agent, agent2: Agent,
                                         context: Optional[Dict] = None) -> Conversation:
        """Execute a single conversation asynchronously."""
        # Plan the interaction (synchronous, fast)
        plan = self.planner.plan_interaction(agent1, agent2, context)
        
        # Execute the conversation asynchronously
        transcript = await self._execute_conversation_async(agent1, agent2, plan)
        
        # Analyze results (can be async in future)
        analysis = self.updater.analyze_conversation(transcript, agent1, agent2, plan)
        
        # Calculate state changes
        state_changes = self.sync_orchestrator._calculate_state_changes(agent1, agent2, analysis)
        
        # Apply updates to agents
        self.sync_orchestrator._apply_updates(agent1, agent2, state_changes)
        
        # Create conversation object
        conversation = Conversation(
            participants=[agent1.id, agent2.id],
            plan=plan,
            transcript=transcript,
            analysis=analysis,
            state_changes=state_changes,
            duration_turns=len(transcript)
        )
        
        # Print summary (synchronous)
        self.sync_orchestrator._print_conversation_summary(agent1, agent2, conversation)
        
        return conversation
    
    async def _execute_conversation_async(self, agent1: Agent, agent2: Agent,
                                        plan: 'InteractionPlan') -> List[Dict]:
        """Run the conversation using async LLM calls."""
        # Create initial prompts (these now include writing style)
        prompt1 = self.sync_orchestrator._create_conversation_prompt(
            agent1, plan.intents[agent1.id], plan)
        prompt2 = self.sync_orchestrator._create_conversation_prompt(
            agent2, plan.intents[agent2.id], plan)
        
        transcript = []
        conversation_context = []
        
        # Determine first speaker
        if agent1.personality.extraversion > agent2.personality.extraversion:
            current_speaker = 1
        elif agent2.personality.extraversion > agent1.personality.extraversion:
            current_speaker = 2
        else:
            current_speaker = 1 if agent1.emotional_state.anxiety > agent2.emotional_state.anxiety else 2
        
        # Conversation loop
        for turn_num in range(min(self.max_turns, plan.expected_duration * 2)):
            if current_speaker == 1:
                # Agent 1 speaks
                message = await self._generate_message_async(
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
                
                if self.sync_orchestrator._is_conversation_complete(message, transcript, plan):
                    break
                
                current_speaker = 2
                
            else:
                # Agent 2 speaks
                message = await self._generate_message_async(
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
                
                if self.sync_orchestrator._is_conversation_complete(message, transcript, plan):
                    break
                
                current_speaker = 1
        
        return transcript
    
    async def _generate_message_async(self, agent: Agent, base_prompt: str,
                                    context: List[Dict], plan: 'InteractionPlan') -> str:
        """Generate a single message asynchronously."""
        # Build conversation history
        history = "\n".join([
            f"{msg['speaker']}: {msg['content']}" 
            for msg in context[-10:]
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
        
        # Generate response asynchronously
        try:
            response = await self.async_llm.generate(full_prompt, temperature=0.8, max_tokens=200)
            
            # Clean up response
            response = response.strip()
            
            # Ensure response isn't too long
            sentences = response.split('. ')
            if len(sentences) > 5:
                max_sentences = 3 if agent.personality.extraversion < 40 else 5
                response = '. '.join(sentences[:max_sentences]) + '.'
            
            return response
            
        except Exception as e:
            logger.warning(f"Async message generation failed: {e}")
            return self.sync_orchestrator._generate_fallback_message(agent, plan)
    
    def cleanup(self):
        """Cleanup resources."""
        if self.owns_wrapper and hasattr(self.async_llm, 'close'):
            self.async_llm.close()