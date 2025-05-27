# Interactions Module Documentation

## Overview

The interactions module manages all aspects of agent conversations in DynaVox, from pre-conversation planning through execution to post-conversation state updates. It implements a sophisticated system that considers personality traits, emotional states, and conversation dynamics to create realistic social interactions.

## Module Structure

```mermaid
classDiagram
    class InteractionPlan {
        +List~String~ topics
        +Dict~String,String~ intents
        +Dict context
        +int expected_duration
    }
    
    class InteractionPlanner {
        +PersonalityBehaviorOntology ontology
        +plan_interaction(agent1, agent2, context)
        -_score_topics(agent1, agent2, shared_topics)
        -_select_topics(topic_scores, context)
        -_determine_intent(agent, partner, topics)
        -_estimate_duration(agent1, agent2)
    }
    
    class Conversation {
        +List~String~ participants
        +InteractionPlan plan
        +List~Dict~ transcript
        +ConversationAnalysis analysis
        +Dict~StateChange~ state_changes
        +DateTime timestamp
        +int duration_turns
    }
    
    class ConversationOrchestrator {
        +LLMClient llm
        +PersonalityBehaviorOntology ontology
        +PromptTemplates prompts
        +InteractionPlanner planner
        +StateUpdater updater
        +int max_turns
        +conduct_conversation(agent1, agent2, context)
        -_execute_conversation(agent1, agent2, plan)
        -_generate_agent_message(agent, partner, transcript, plan)
        -_should_continue_conversation(transcript, plan, turn_count)
        +_print_conversation_summary(agent1, agent2, conversation)
    }
    
    class AsyncConversationOrchestrator {
        +AsyncLLMWrapper async_llm
        +ConversationOrchestrator sync_orchestrator
        +conduct_conversations_parallel(interaction_pairs, context)
        -_conduct_single_conversation(agent1, agent2, context, index)
        -_execute_conversation_async(agent1, agent2, plan)
        -_generate_messages_batch(message_requests)
    }
    
    class ConversationAnalysis {
        +List~String~ topics_discussed
        +AgentPerspective agent1_perspective
        +AgentPerspective agent2_perspective
        +float overall_quality
        +bool resolution_achieved
    }
    
    class AgentPerspective {
        +List~String~ topics_discussed
        +Dict~List~ arguments_made
        +Dict~List~ arguments_encountered
        +float interaction_quality
        +float validation_received
        +float conflict_level
    }
    
    class StateChange {
        +Dict~Dict~ opinion_changes
        +Dict~float~ emotion_changes
    }
    
    class StateUpdater {
        +LLMClient llm
        +PersonalityBehaviorOntology ontology
        +PromptTemplates prompts
        +analyze_conversation(transcript, agent1, agent2, plan)
        +calculate_state_changes(agent, partner, analysis, perspective)
        +apply_state_changes(agent, changes)
        -_calculate_influence(agent, partner, topic, arguments, quality)
        -_calculate_emotional_impact(agent, quality, validation, conflict)
        -_assess_credibility(agent, partner, topic)
        -_evaluate_argument_quality(arguments, agent, topic)
    }
    
    class EnhancedStateUpdater {
        +calculate_state_changes(agent, partner, analysis, perspective)
        -_classify_conversation_type(validation, conflict, quality)
        -_calculate_enhanced_influence(...)
        -_calculate_enhanced_emotional_impact(...)
        +apply_state_changes(agent, changes)
    }
    
    ConversationOrchestrator --> InteractionPlanner : uses
    ConversationOrchestrator --> StateUpdater : uses
    ConversationOrchestrator --> Conversation : creates
    AsyncConversationOrchestrator --> ConversationOrchestrator : wraps
    StateUpdater --> ConversationAnalysis : produces
    StateUpdater --> StateChange : produces
    EnhancedStateUpdater --|> StateUpdater : extends
    InteractionPlanner --> InteractionPlan : creates
    ConversationAnalysis --> AgentPerspective : contains
```

## Conversation Flow

The complete conversation process follows this sequence:

```mermaid
sequenceDiagram
    participant Engine as SimulationEngine
    participant Planner as InteractionPlanner
    participant Orchestrator as ConversationOrchestrator
    participant LLM as LLMClient
    participant Updater as StateUpdater
    participant Agent1
    participant Agent2
    
    Engine->>Orchestrator: conduct_conversation(agent1, agent2)
    
    Note over Orchestrator: Pre-Conversation Planning
    Orchestrator->>Planner: plan_interaction(agent1, agent2)
    Planner->>Planner: _score_topics()
    Planner->>Planner: _select_topics()
    Planner->>Planner: _determine_intent()
    Planner->>Planner: _estimate_duration()
    Planner-->>Orchestrator: InteractionPlan
    
    Note over Orchestrator: Conversation Execution
    loop Until conversation ends
        Orchestrator->>LLM: Generate agent1 message
        LLM-->>Orchestrator: Message text
        Orchestrator->>Agent1: Update transcript
        
        Orchestrator->>LLM: Generate agent2 message
        LLM-->>Orchestrator: Message text
        Orchestrator->>Agent2: Update transcript
        
        Orchestrator->>Orchestrator: _should_continue_conversation()
    end
    
    Note over Orchestrator: Post-Conversation Analysis
    Orchestrator->>Updater: analyze_conversation(transcript)
    Updater->>LLM: ANALYZE_CONVERSATION prompt
    LLM-->>Updater: Analysis JSON
    
    Note over Orchestrator: State Updates
    Updater->>Updater: calculate_state_changes(agent1)
    Updater->>Agent1: apply_state_changes()
    
    Updater->>Updater: calculate_state_changes(agent2)
    Updater->>Agent2: apply_state_changes()
    
    Orchestrator->>Orchestrator: _print_conversation_summary()
    Orchestrator-->>Engine: Conversation object
```

## Core Components

### 1. Interaction Planning (`planner.py`)

The `InteractionPlanner` determines what agents will discuss and why:

#### Topic Selection Process

```mermaid
graph TD
    A[Shared Topics] --> B[Score Topics]
    B --> C{Scoring Factors}
    C --> D[Importance]
    C --> E[Emotional Charge]
    C --> F[Position Difference]
    C --> G[Knowledge Levels]
    C --> H[Emotional States]
    
    D --> I[Base Score]
    E --> I
    F --> J[Disagreement Bonus]
    G --> K[Low Knowledge Penalty]
    H --> L[Anxiety Modifiers]
    
    I --> M[Final Score]
    J --> M
    K --> M
    L --> M
    
    M --> N[Sort by Score]
    N --> O[Select Top Topics]
    O --> P[Selected Topics]
```

#### Intent Determination

Conversation intents are personality-driven:
- **Learn**: High openness, low emotionality
- **Persuade**: High extraversion, low agreeableness
- **Validate**: High emotionality, low openness
- **Bond**: High agreeableness, high extraversion
- **Debate**: High openness, low agreeableness
- **Explore**: High openness, high conscientiousness

### 2. Conversation Orchestration (`orchestrator.py`)

The `ConversationOrchestrator` manages the conversation execution:

#### Message Generation
Each message is generated considering:
- Agent's full profile and current state
- Conversation history
- Assigned intent
- Personality-based behavior patterns
- Current emotional state

#### Conversation Termination
Conversations end when:
- Maximum turns reached
- Natural conclusion keywords detected
- Social energy depleted
- High cognitive load for both agents

#### Summary Generation
After each conversation:
- Brief 50-word summary generated
- Opinion changes displayed with directional indicators
- Emotional changes shown with appropriate emojis
- Interaction quality assessed

### 3. Asynchronous Orchestration (`async_orchestrator.py`)

Enables parallel conversation execution:

```mermaid
graph LR
    A[Interaction Pairs] --> B{Async Orchestrator}
    B --> C[Conversation 1]
    B --> D[Conversation 2]
    B --> E[Conversation 3]
    B --> F[Conversation N]
    
    C --> G[Parallel Execution]
    D --> G
    E --> G
    F --> G
    
    G --> H[Batch LLM Calls]
    H --> I[Completed Conversations]
```

#### Parallel Processing
- Uses `AsyncLLMWrapper` with ThreadPoolExecutor
- Batches multiple conversations for efficiency
- Shows real-time status updates
- Falls back to sequential execution on failure

### 4. State Updates (`updater.py`)

The `StateUpdater` analyzes conversations and calculates state changes:

#### Influence Calculation

```mermaid
graph TD
    A[Influence Factors] --> B[Base Influence]
    
    C[Source Credibility] --> B
    D[Argument Quality] --> B
    E[Social Influence] --> B
    
    B --> F[Personality Moderation]
    G[Openness Modifier] --> F
    H[Certainty Barrier] --> F
    I[Evidence Need] --> F
    
    F --> J[Effective Influence]
    K[Interaction Quality] --> J
    
    J --> L[Position Change]
    J --> M[Certainty Change]
    J --> N[Importance Change]
```

#### Key Influence Factors:

1. **Source Credibility** (0.0-1.0):
   - Education level comparison
   - Knowledge differential
   - Personality-based trust
   - Professional relevance

2. **Argument Quality** (0.0-1.0):
   - LLM evaluation of arguments
   - Agent's conscientiousness affects requirements
   - Openness affects receptivity

3. **Social Influence** (0.0-1.0):
   - High agreeableness + good interaction → influence
   - Empathy-driven opinion change
   - Conflict avoidance motivations

### 5. Enhanced State Updates (`updater_enhanced.py`)

The `EnhancedStateUpdater` adds sophisticated conversation dynamics:

#### Conversation Type Classification

```mermaid
stateDiagram-v2
    [*] --> Analysis
    Analysis --> EchoChamber: High Validation, Low Conflict
    Analysis --> HeatedDebate: Low Validation, High Conflict
    Analysis --> RespectfulDisagreement: Moderate Both
    Analysis --> ProductiveExchange: High Quality
    Analysis --> NeutralConversation: Default
    
    EchoChamber --> [*]: Increase Certainty & Extremity
    HeatedDebate --> [*]: Increase Importance, Variable Position
    RespectfulDisagreement --> [*]: Enable Position Change
    ProductiveExchange --> [*]: Balanced Changes
    NeutralConversation --> [*]: Minimal Changes
```

#### Enhanced Dynamics by Type:

1. **Echo Chamber**:
   - Same-side validation → increased extremity
   - Certainty increases significantly
   - Importance increases moderately
   - Opposite-side validation → cognitive dissonance

2. **Heated Debate**:
   - Disagreeable personalities → entrenchment
   - Agreeable personalities → larger shifts
   - Importance increases dramatically
   - High emotional charge

3. **Respectful Disagreement**:
   - Most conducive to genuine change
   - Moderate certainty reduction
   - Increased importance through engagement

4. **Productive Exchange**:
   - Quality arguments increase certainty
   - Balanced position changes
   - Knowledge increases

## Opinion Evolution Mechanics

### Position Changes

```python
# Base calculation
position_diff = partner.position - agent.position
position_delta = position_diff * effective_influence * 0.15

# Enhanced modifiers
if conversation_type == "echo_chamber" and same_side:
    position_delta = sign(current_position) * 5 * validation_level
elif conversation_type == "heated_debate":
    if agent.agreeableness < 40:  # Backlash effect
        position_delta = -position_delta * 0.5
    else:
        position_delta *= 2.0
```

### Certainty Evolution

- Increases with: validation, good arguments, echo chambers
- Decreases with: strong counter-arguments, cognitive dissonance
- Personality moderation via conscientiousness

### Importance Changes

- Conflict always increases importance
- Emotional engagement increases importance
- Validation in echo chambers increases importance
- High arousal states amplify importance changes

## Emotional Impact

### Base Emotional Changes

```mermaid
graph TD
    A[Interaction Factors] --> B[Emotional Changes]
    
    C[Interaction Quality] --> D[Valence Change]
    E[Validation Received] --> D
    F[Conflict Level] --> D
    
    F --> G[Anxiety Change]
    C --> G
    E --> G
    
    E --> H[Confidence Change]
    C --> H
    F --> H
    
    I[Personality] --> J[Social Energy Change]
    C --> J
    F --> J
    
    F --> K[Cognitive Load Change]
    L[Topics Discussed] --> K
```

### Enhanced Emotional Dynamics

By conversation type:
- **Echo Chamber**: +8 valence, +10 confidence, +5 social energy
- **Heated Debate**: -5 valence, +10 anxiety, +15 cognitive load
- **Respectful Disagreement**: +3 valence (if open), +10 cognitive load
- **Productive Exchange**: +5 valence, +5 confidence, +5 cognitive load

## Professional Topic Relevance

The system maintains detailed mappings of which professions have expertise in which topics:

- **Climate Change**: Environmental scientists, farmers, engineers, educators
- **Healthcare**: Medical professionals, insurance workers, administrators
- **Remote Work**: Tech workers, creatives, consultants, educators
- **AI Regulation**: Tech professionals, lawyers, ethicists, affected workers
- **Wealth Inequality**: Economists, social workers, service industry, academics

Professional relevance increases source credibility by 15%.

## Usage Patterns

### Single Conversation
```python
orchestrator = ConversationOrchestrator(llm_client)
conversation = orchestrator.conduct_conversation(agent1, agent2, context)
```

### Parallel Conversations
```python
async_orchestrator = AsyncConversationOrchestrator(llm_client)
conversations = await async_orchestrator.conduct_conversations_parallel(
    interaction_pairs, context
)
```

### Custom State Updates
```python
updater = EnhancedStateUpdater(llm_client)
state_changes = updater.calculate_state_changes(
    agent, partner, analysis, perspective
)
updater.apply_state_changes(agent, state_changes)
```

## Key Design Principles

1. **Personality-Driven**: All aspects of conversation flow from HEXACO traits
2. **Multi-Factor Influence**: Opinion changes consider many realistic factors
3. **Emotional Realism**: Emotional states affect and are affected by conversations
4. **Conversation Dynamics**: Different conversation types produce different outcomes
5. **Professional Expertise**: Domain knowledge affects credibility
6. **Graceful Degradation**: Fallback mechanisms for LLM failures