# Simulation Module Documentation

## Overview

The simulation module is the core engine of DynaVox, orchestrating the entire agent-based social dynamics simulation. It manages agent population creation, interaction rounds, conversation scheduling, metrics calculation, and result analysis. The module supports both synchronous and asynchronous execution modes for scalability.

## Module Structure

```mermaid
classDiagram
    class SimulationEngine {
        +LLMClient llm_client
        +AgentGenerator generator
        +ConversationOrchestrator orchestrator
        +AsyncConversationOrchestrator async_orchestrator
        +SimulationAnalyzer analyzer
        +Dict~Agent~ agents
        +List~Conversation~ conversations
        +List~PopulationMetrics~ metrics_history
        +int round_number
        +RealTimeDataWriter data_writer
        +bool use_async
        +initialize_population(size, topics, demographics, personality_distribution)
        +run_interaction_round(interaction_probability, homophily_bias, max_interactions)
        +run_simulation(rounds, interaction_probability, homophily_bias)
        +export_results(output_dir)
        -_sample_personality_bias(distribution, index)
        -_calculate_similarity(agent1, agent2)
        -_calculate_interaction_probability(agent1, agent2, base_prob, homophily_bias)
        -_execute_conversations_async(interaction_pairs)
        -_execute_conversations_sync(interaction_pairs)
    }
    
    class SimulationAnalyzer {
        +calculate_population_metrics(agents, conversations, round_number)
        +identify_influencers(conversations, agents)
        +analyze_opinion_trajectories(metrics_history, topic)
        +find_echo_chambers(agents, conversations)
        +generate_cluster_report(metrics_history)
        -_calculate_topic_metrics(agents, topic)
        -_calculate_polarization(positions)
        -_calculate_consensus(positions)
        -_identify_clusters_detailed(positions, agent_ids, agents, topic)
        -_calculate_opinion_similarity(agent1, agent2)
        -_describe_cluster_position(mean_position, topic)
    }
    
    class PopulationMetrics {
        +DateTime timestamp
        +int round_number
        +Dict~Dict~ opinion_metrics
        +float overall_polarization
        +float overall_consensus
        +float avg_certainty
        +float avg_emotional_valence
        +int interaction_count
        +float avg_interaction_quality
    }
    
    SimulationEngine --> AgentGenerator : uses
    SimulationEngine --> ConversationOrchestrator : uses
    SimulationEngine --> AsyncConversationOrchestrator : uses
    SimulationEngine --> SimulationAnalyzer : uses
    SimulationAnalyzer --> PopulationMetrics : creates
```

## Simulation Flow

The complete simulation process follows this sequence:

```mermaid
sequenceDiagram
    participant User
    participant Engine as SimulationEngine
    participant Generator as AgentGenerator
    participant Analyzer as SimulationAnalyzer
    participant Orchestrator as ConversationOrchestrator
    
    User->>Engine: run_simulation(rounds)
    
    Note over Engine: Phase 1: Population Initialization
    Engine->>Engine: initialize_population()
    loop For each agent
        Engine->>Generator: generate_agent(constraints)
        Generator-->>Engine: Agent
    end
    
    Note over Engine: Phase 2: Simulation Rounds
    loop For each round
        Engine->>Engine: run_interaction_round()
        Note over Engine: Pair Selection
        Engine->>Engine: Form n/2 potential pairs
        Engine->>Engine: Calculate interaction probabilities
        Engine->>Engine: Select actual interaction pairs
        
        Note over Engine: Conversation Execution
        alt Async Mode
            Engine->>Engine: _execute_conversations_async()
        else Sync Mode
            loop For each pair
                Engine->>Orchestrator: conduct_conversation()
                Orchestrator-->>Engine: Conversation
            end
        end
        
        Note over Engine: Metrics Calculation
        Engine->>Analyzer: calculate_population_metrics()
        Analyzer-->>Engine: PopulationMetrics
    end
    
    Note over Engine: Phase 3: Analysis & Export
    Engine->>Analyzer: identify_influencers()
    Engine->>Analyzer: find_echo_chambers()
    Engine->>Analyzer: generate_cluster_report()
    Engine->>Engine: export_results()
    
    Engine-->>User: Simulation Results
```

## Core Components

### 1. Simulation Engine (`engine.py`)

The main coordinator for the entire simulation:

#### Population Initialization

```mermaid
graph TD
    A[initialize_population] --> B[Define Constraints]
    B --> C{Personality Distribution}
    C -->|Provided| D[Use Custom Distribution]
    C -->|Default| E[Use Diverse Biases]
    
    E --> F[High Openness]
    E --> G[High Conscientiousness]
    E --> H[High Agreeableness]
    E --> I[High Extraversion]
    E --> J[Balanced]
    E --> K[Low Traits]
    
    D --> L[Generate Agent]
    F --> L
    G --> L
    H --> L
    I --> L
    J --> L
    K --> L
    
    L --> M{Success?}
    M -->|Yes| N[Add to Population]
    M -->|No| O[Retry with Defaults]
    O --> P{Success?}
    P -->|Yes| N
    P -->|No| Q[Skip Agent]
```

Default personality biases for diversity:
- High openness
- High conscientiousness
- High agreeableness
- High extraversion
- Balanced
- Low agreeableness
- High emotionality
- Low openness
- Low conscientiousness
- Low extraversion
- Various professional biases (analytical, creative, practical)

#### Interaction Round Mechanics

Each round follows a sophisticated pairing and selection process:

```mermaid
graph TD
    A[Start Round] --> B[Shuffle Agents]
    B --> C[Create n/2 Potential Pairs]
    
    C --> D[Calculate Interaction Probabilities]
    D --> E{For Each Pair}
    
    E --> F[Base Probability]
    E --> G[Calculate Similarity]
    E --> H[Apply Homophily Bias]
    
    F --> I[Adjusted Probability]
    G --> I
    H --> I
    
    I --> J{Random < Probability?}
    J -->|Yes| K[Add to Interaction List]
    J -->|No| L[Skip Pair]
    
    K --> M[Execute Conversations]
    L --> M
    
    M --> N{Async Mode?}
    N -->|Yes| O[Parallel Execution]
    N -->|No| P[Sequential Execution]
```

##### Similarity Calculation

Agents' similarity is calculated across multiple dimensions:

```python
similarity = weighted_average([
    personality_similarity * 0.3,
    opinion_similarity * 0.4,
    background_similarity * 0.2,
    emotional_similarity * 0.1
])
```

##### Homophily-Adjusted Probability

```python
# High homophily (>0.5): Similar agents more likely to interact
if homophily_bias > 0.5:
    adjusted_prob = base_prob * (0.5 + similarity * homophily_strength)
# Low homophily (<0.5): Different agents more likely to interact  
else:
    adjusted_prob = base_prob * (0.5 + (1 - similarity) * homophily_strength)
```

#### Conversation Execution Modes

##### Synchronous Mode
- Conversations execute one at a time
- Simpler debugging and monitoring
- Suitable for smaller populations

##### Asynchronous Mode
- All conversations in a round execute in parallel
- Uses ThreadPoolExecutor for concurrent LLM calls
- Visual status indicators during execution
- Significantly faster for large populations

### 2. Simulation Analyzer (`analyzer.py`)

Provides comprehensive analysis of simulation dynamics:

#### Population Metrics Calculation

```mermaid
graph LR
    A[Agent Population] --> B[Topic Metrics]
    A --> C[Emotional Metrics]
    A --> D[Interaction Metrics]
    
    B --> E[Mean Position]
    B --> F[Polarization]
    B --> G[Consensus]
    B --> H[Clusters]
    
    C --> I[Avg Valence]
    C --> J[Avg Certainty]
    
    D --> K[Interaction Count]
    D --> L[Avg Quality]
    
    E --> M[Population Metrics]
    F --> M
    G --> M
    H --> M
    I --> M
    J --> M
    K --> M
    L --> M
```

#### Opinion Clustering Algorithm

The analyzer identifies opinion clusters using a gap-based approach:

```mermaid
graph TD
    A[Sort Positions] --> B[Find Gaps > 30]
    B --> C[Define Cluster Boundaries]
    C --> D[Calculate Cluster Stats]
    
    D --> E[Mean Position]
    D --> F[Size & Percentage]
    D --> G[Sample Agents]
    D --> H[Description]
    
    E --> I[Cluster Summary]
    F --> I
    G --> I
    H --> I
```

Cluster descriptions are generated based on mean position:
- `> 50`: "strongly support"
- `20 to 50`: "moderately support"
- `-20 to 20`: "are neutral about"
- `-50 to -20`: "moderately oppose"
- `< -50`: "strongly oppose"

#### Polarization Measurement

Polarization is calculated as the bimodality of opinion distribution:

```python
# Find histogram peaks
peaks = find_local_maxima(opinion_histogram)

if len(peaks) >= 2:
    # Distance between two highest peaks
    distance = abs(peak1_pos - peak2_pos) / 200
    
    # Balance of peak heights
    height_ratio = min(peak1_height, peak2_height) / max(peak1_height, peak2_height)
    
    # Polarization combines distance and balance
    polarization = distance * height_ratio
```

#### Influencer Identification

Agents are ranked by total opinion change they cause in others:

```python
for conversation in conversations:
    for agent_id, changes in conversation.state_changes.items():
        partner_id = get_conversation_partner(agent_id)
        influence_scores[partner_id] += sum_absolute_opinion_changes(changes)
```

#### Echo Chamber Detection

Identifies groups that:
1. Interact frequently (≥3 times)
2. Have high opinion similarity (>0.7)
3. Form clusters of ≥3 agents

### 3. Real-Time Data Export

The engine supports real-time data writing for monitoring:

```python
if self.data_writer:
    # After agent creation
    self.data_writer.update_agent(agent)
    
    # After conversation
    self.data_writer.add_conversation(conversation)
    
    # After metrics calculation
    self.data_writer.add_metrics(metrics)
```

## Configuration Options

### Population Parameters
- `size`: Number of agents (typical: 10-200)
- `topics`: List of opinion topics
- `demographics`: Constraints on age, occupation, etc.
- `personality_distribution`: Custom trait distributions

### Interaction Parameters
- `interaction_probability`: Base chance of interaction (0.0-1.0)
- `homophily_bias`: Preference for similar agents (0.0-1.0)
  - 0.0: Strong preference for different agents
  - 0.5: No preference (random)
  - 1.0: Strong preference for similar agents
- `max_interactions_per_agent`: Limit per round

### Execution Parameters
- `rounds`: Number of simulation rounds
- `use_async`: Enable parallel conversation execution
- `seed`: Random seed for reproducibility

## Output Structure

```
output_dir/
├── data/
│   ├── agents.json          # Final agent states
│   ├── conversations.json   # All conversations
│   └── metrics_history.json # Metrics by round
├── visualizations/
│   ├── opinion_evolution_*.png
│   ├── polarization_*.png
│   └── interaction_network.png
├── reports/
│   ├── simulation_report.txt
│   ├── cluster_analysis.json
│   └── influencer_report.csv
├── checkpoints/
│   └── round_*.json
└── analysis.json           # Key findings
```

## Key Algorithms

### 1. Pairing Algorithm (n/2 pairs)

```python
def create_potential_pairs(agents):
    shuffled = random.shuffle(agents)
    pairs = []
    
    # Create n/2 pairs from shuffled list
    for i in range(0, len(shuffled)-1, 2):
        pairs.append((shuffled[i], shuffled[i+1]))
    
    # Fill remaining with random pairs if needed
    while len(pairs) < target_pairs:
        pair = random.sample(unused_agents, 2)
        pairs.append(pair)
    
    return pairs
```

### 2. Similarity Calculation

```python
def calculate_similarity(agent1, agent2):
    # Personality similarity (HEXACO traits)
    personality_sim = 1 - euclidean_distance(
        agent1.personality, agent2.personality
    ) / max_distance
    
    # Opinion similarity
    shared_topics = set(agent1.opinions) & set(agent2.opinions)
    opinion_distances = [
        abs(agent1.opinions[t].position - agent2.opinions[t].position) / 200
        for t in shared_topics
    ]
    opinion_sim = 1 - mean(opinion_distances)
    
    # Background similarity
    background_sim = calculate_tag_overlap(
        agent1.background, agent2.background
    )
    
    # Emotional state similarity
    emotional_sim = 1 - abs(
        agent1.emotional_state.valence - agent2.emotional_state.valence
    ) / 100
    
    # Weighted combination
    return (personality_sim * 0.3 + opinion_sim * 0.4 + 
            background_sim * 0.2 + emotional_sim * 0.1)
```

## Usage Example

```python
from src.simulation import SimulationEngine
from src.llm import create_llm_client

# Initialize
llm_client = create_llm_client("gpt-4o-mini")
engine = SimulationEngine(llm_client, seed=42, use_async=True)

# Create population
engine.initialize_population(
    size=50,
    topics=['climate_change', 'ai_regulation', 'wealth_inequality'],
    demographics={'age_range': [25, 65]}
)

# Run simulation
engine.run_simulation(
    rounds=10,
    interaction_probability=0.2,
    homophily_bias=0.6
)

# Export results
engine.export_results("output/my_simulation")

# Access analysis
print(f"Final polarization: {engine.metrics_history[-1].overall_polarization}")
influencers = engine.analyzer.identify_influencers(
    engine.conversations, engine.agents
)
```

## Performance Considerations

1. **Async Mode**: Use for populations > 20 agents
2. **Batch Size**: Async mode processes all round conversations in parallel
3. **Memory**: ~10MB per 100 agents with full history
4. **LLM Costs**: See [LLM module](llm.md) for cost estimation
5. **Checkpointing**: Automatic every 5 rounds for recovery

## Key Design Principles

1. **Modularity**: Clear separation between generation, interaction, and analysis
2. **Scalability**: Async support for large populations
3. **Reproducibility**: Seed support for deterministic simulations
4. **Observability**: Real-time data export and comprehensive metrics
5. **Flexibility**: Configurable parameters for different research questions