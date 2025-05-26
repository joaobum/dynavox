# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DynaVox is an agent-based social dynamics modeling framework where each agent is powered by Large Language Models (LLMs) with detailed psychological profiles. The framework models opinion change through natural conversations between psychologically consistent agents based on the HEXACO personality model.

## Core Architecture

The framework uses a two-tier architecture:

### Traits Layer (Immutable)
- **Personality**: HEXACO model (6 traits, 0-100 scale)
- **Background**: Name, demographics, education, occupation, cultural factors
- **Emotional Baseline**: Stable emotional tendencies
- **Biography**: Rich 800-1200 word narrative synthesizing all traits

### State Layer (Dynamic)
- **Emotional State**: Current mood, anxiety, confidence, social energy, cognitive load
- **Opinions**: Multi-dimensional (position, certainty, importance, knowledge, emotional charge)

## Latest Features

### Enhanced Conversation System
- **Personality-based writing styles**: Agents speak according to their personality traits
- **Conversation summaries**: 50-word summaries after each interaction
- **Opinion/emotion tracking**: Detailed change reports after conversations
- **Enhanced prompts**: Better definitions for interaction quality and argument evaluation

### Parallel Execution
- **Async conversation support**: Multiple conversations can run concurrently
- **n/2 pairing system**: Each round forms n/2 potential conversation pairs
- **Probability-based interaction**: Pairs evaluate whether to actually converse

### Improved Diversity
- **Name generation**: Culturally diverse names based on background
- **Expanded topics**: 15+ opinion topics available
- **Professional relevance**: Rich mapping of topics to professions

### Analysis Enhancements
- **Cluster analysis**: Detailed opinion cluster summaries per topic
- **Group dynamics**: Track how opinion groups evolve over time
- **Influence tracking**: Identify most influential agents

## Development Commands

### Installation
```bash
# Install with virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .

# For specific LLM providers
pip install -e ".[openai]"    # OpenAI support
pip install -e ".[anthropic]"  # Anthropic support
pip install -e ".[analysis]"   # Analysis tools
pip install -e ".[dev]"        # Development tools
```

### Running Simulations
```bash
# List available models
python example_simulation.py --list-models

# Run with specific model
python example_simulation.py --model gpt-4o-mini  # Most cost-effective
python example_simulation.py --model gpt-4o       # Higher quality
python example_simulation.py --model gpt-3.5-turbo

# Run with mock LLM (no API calls)
python example_simulation.py --mock

# Enable async parallel conversations
python example_simulation.py --async

# Combine options
python example_simulation.py --mock --async

# Analyze existing results
python example_simulation.py --analyze-only
```

### Testing
```bash
# Run tests (when implemented)
pytest

# Run with coverage
pytest --cov=src

# Quick test with mock
python example_simulation.py --mock
```

### Code Quality
```bash
# Format code
black src/

# Lint code
flake8 src/

# Type checking
mypy src/
```

## Project Structure

```
DynaVox/
├── src/
│   ├── agents/
│   │   ├── generator.py      # Agent generation with LLMs
│   │   ├── name_generator.py # Culturally diverse name generation
│   │   ├── personality.py    # HEXACO behavior mappings
│   │   └── profile.py        # Agent data structures
│   ├── interactions/
│   │   ├── orchestrator.py       # Conversation management
│   │   ├── async_orchestrator.py # Parallel conversation execution
│   │   ├── planner.py            # Pre-conversation planning
│   │   └── updater.py            # Post-conversation state updates
│   ├── llm/
│   │   ├── client.py        # LLM provider interfaces
│   │   ├── async_client.py  # Async LLM interfaces
│   │   └── prompts.py       # Conversation and analysis prompts
│   ├── simulation/
│   │   ├── engine.py        # Main simulation loop
│   │   └── analyzer.py      # Metrics and analysis
│   └── config.py            # Configuration constants
├── example_simulation.py    # Example usage script
└── docs/
    ├── design-spec.md       # Original design document
    ├── implementation-spec.md # Technical specification
    └── todo.md              # Completed feature list
```

## Key Implementation Details

### Agent Generation Flow
1. Generate personality traits using LLM with psychological consistency
2. Generate culturally appropriate name based on background
3. Generate background consistent with personality
4. Derive emotional baseline from personality and background
5. Generate comprehensive biography that weaves all elements together
6. Initialize opinions based on full agent profile
7. Set initial emotional state

### Conversation System
The conversation flow involves:
1. **Pre-Interaction Planning**: 
   - Form n/2 potential conversation pairs each round
   - Evaluate interaction probability based on similarity and social energy
   - Select topics based on importance and emotional charge
   - Determine conversation intent based on personality
2. **Conversation Execution**: 
   - Each agent embodied by LLM with full profile prompt
   - Personality-based writing styles guide speech patterns
   - Natural turn-taking with personality-driven behavior
   - Async execution for parallel conversations
3. **Post-Interaction Updates**: 
   - Opinion changes based on personality-moderated influence
   - Emotional state evolution based on interaction quality
   - Detailed change reporting with summaries

### Personality-Behavior Ontology
The `PersonalityBehaviorOntology` class in `src/agents/personality.py` defines systematic mappings between HEXACO traits and behavioral tendencies, including:
- Conversation patterns (communication style, topic approach, conflict style)
- Influence modifiers (how personality affects receptivity to opinion change)
- Intent probabilities (likelihood of different conversation goals)
- Writing style guidelines (verbosity, vocabulary, emotional expression)

### LLM Integration
- Multiple provider support through abstract base class
- Async client support for parallel execution
- Model selection with cost considerations
- Carefully crafted prompts in `src/llm/prompts.py` ensure consistent agent behavior
- Enhanced prompts for better interaction quality assessment

## Configuration

The `src/config.py` file contains all configurable parameters including:
- Available LLM models and their costs (updated for 2024)
- Conversation parameters (max turns, conclusion keywords)
- Influence mechanics (thresholds, max changes)
- Personality and emotion bounds
- Topic definitions (15+ topics with professional relevance)
- Simulation defaults

## Environment Setup

Create a `.env` file with your API keys:
```bash
OPENAI_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here
```

## Output Structure

Simulations generate in `example_results/data/`:
- `agents.json`: Final agent states with full profiles
- `metrics_history.json`: Population metrics over time
- `conversations.json`: Complete conversation logs with analysis
- `analysis.json`: Key findings including:
  - Top influencers
  - Echo chambers
  - Opinion cluster evolution
  - Final population metrics
- `checkpoints/`: Periodic state saves

## Recent Improvements (2024)

1. **Conversation Enhancements**
   - Added personality-based writing styles
   - Print opinion/emotional changes after each interaction
   - Generate conversation summaries

2. **Analysis Improvements**
   - Group opinion cluster summaries per topic
   - Track cluster evolution over time
   - Enhanced interaction quality definitions

3. **Performance & Scale**
   - Async/parallel conversation execution
   - n/2 pairing system for efficient rounds
   - Support for latest LLM models (GPT-4o, Claude 3.5)

4. **Diversity & Realism**
   - Culturally diverse name generation
   - Expanded professional relevance mappings
   - More nuanced argument quality evaluation

## Best Practices

1. **Cost Management**: Use `--mock` for development, `gpt-4o-mini` for testing, higher models for production
2. **Async Usage**: Enable `--async` for simulations with many agents
3. **Topics**: Start with 3-5 topics to keep conversations focused
4. **Population Size**: 10-20 agents for testing, 50-100 for meaningful dynamics
5. **Rounds**: 5-10 rounds typically show interesting evolution