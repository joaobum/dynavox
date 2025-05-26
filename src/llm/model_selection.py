"""Utilities for model selection and configuration."""
from typing import Dict, Optional
import os
from dotenv import load_dotenv
from .client import OpenAIClient, AnthropicClient, MockLLMClient, LLMClient
from ..config import OPENAI_MODELS, DEFAULT_OPENAI_MODEL

# Load environment variables from .env file
load_dotenv()


def get_available_models() -> Dict[str, str]:
    """Get dictionary of all available models across providers."""
    models = {}
    
    # OpenAI models - Updated for 2024/2025
    models["openai:gpt-4o"] = "GPT-4o - Multimodal, fast, cost-effective (128K context)"
    models["openai:gpt-4o-2024-11-20"] = "GPT-4o latest - Most recent GPT-4o version"
    models["openai:gpt-4o-mini"] = "GPT-4o Mini - Most cost-efficient small model with vision"
    models["openai:gpt-4-turbo"] = "GPT-4 Turbo - High capability, 128K context"
    models["openai:gpt-4-turbo-2024-04-09"] = "GPT-4 Turbo April 2024 - Stable version"
    models["openai:gpt-4"] = "GPT-4 - Original, 8K context"
    models["openai:gpt-4-32k"] = "GPT-4 32K - Extended context window"
    models["openai:gpt-3.5-turbo"] = "GPT-3.5 Turbo - Fast and cost-effective"
    models["openai:gpt-3.5-turbo-0125"] = "GPT-3.5 Turbo latest - 16K context, optimized"
    
    # Anthropic models - Updated for 2024/2025
    models["anthropic:claude-opus-4"] = "Claude Opus 4 - Most powerful (200K context)"
    models["anthropic:claude-sonnet-4"] = "Claude Sonnet 4 - Balanced performance (200K context)"
    models["anthropic:claude-3-5-sonnet"] = "Claude 3.5 Sonnet - Upgraded, same price (200K context)"
    models["anthropic:claude-3-5-sonnet-20241022"] = "Claude 3.5 Sonnet Oct 2024 - Latest version"
    models["anthropic:claude-3-5-haiku"] = "Claude 3.5 Haiku - Fast and affordable (200K context)"
    models["anthropic:claude-3-opus-20240229"] = "Claude 3 Opus - Previous generation"
    models["anthropic:claude-3-sonnet-20240229"] = "Claude 3 Sonnet - Previous generation"
    models["anthropic:claude-3-haiku-20240307"] = "Claude 3 Haiku - Previous generation"
    
    # Mock model
    models["mock:test"] = "Mock LLM for testing (no API calls)"
    
    return models


def create_llm_client(model_spec: str = None, 
                     api_key: Optional[str] = None,
                     auto_load_env: bool = True) -> LLMClient:
    """Create an LLM client based on model specification.
    
    Args:
        model_spec: Model specification in format "provider:model" 
                   (e.g., "openai:gpt-4", "anthropic:claude-3-opus")
                   If None, uses DEFAULT_MODEL from env or config
        api_key: API key for the provider (auto-loaded from env if not provided)
        auto_load_env: Whether to automatically load from environment variables
    
    Returns:
        Configured LLM client
        
    Examples:
        >>> client = create_llm_client()  # Uses DEFAULT_MODEL from .env
        >>> client = create_llm_client("gpt-4o")  # Auto-detects OpenAI
        >>> client = create_llm_client("claude-3-5-sonnet")  # Auto-detects Anthropic
        >>> client = create_llm_client("mock:test")  # For testing
    """
    # Check if we should use mock LLM
    if auto_load_env and os.getenv("USE_MOCK_LLM", "false").lower() == "true":
        print("Using Mock LLM (USE_MOCK_LLM=true in environment)")
        return MockLLMClient()
    
    # Get default model from environment or config
    if model_spec is None:
        model_spec = os.getenv("DEFAULT_MODEL", f"openai:{DEFAULT_OPENAI_MODEL}")
    
    # Auto-detect provider if not specified
    if ":" not in model_spec:
        # Check if it's a known OpenAI model
        openai_models = ["gpt-4", "gpt-4o", "gpt-3.5", "gpt-4-turbo"]
        if any(model_spec.startswith(prefix) for prefix in openai_models):
            provider = "openai"
            model = model_spec
        # Check if it's a known Anthropic model
        elif "claude" in model_spec.lower():
            provider = "anthropic"
            model = model_spec
        else:
            # Default to OpenAI
            provider = "openai"
            model = model_spec
    else:
        provider, model = model_spec.split(":", 1)
    
    provider = provider.lower()
    
    # Auto-load API key from environment if not provided
    if api_key is None and auto_load_env:
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment. "
                               "Please set it in .env file or pass explicitly.")
        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment. "
                               "Please set it in .env file or pass explicitly.")
    
    # Create appropriate client
    if provider == "openai":
        return OpenAIClient(api_key=api_key, model=model)
    elif provider == "anthropic":
        return AnthropicClient(api_key=api_key, model=model)
    elif provider == "mock":
        return MockLLMClient()
    else:
        raise ValueError(f"Unknown provider: {provider}. "
                        f"Available: openai, anthropic, mock")


def estimate_cost(model_spec: str, num_agents: int, 
                 num_rounds: int, interactions_per_round: int) -> Dict[str, float]:
    """Estimate approximate cost for a simulation.
    
    Args:
        model_spec: Model specification (e.g., "openai:gpt-4")
        num_agents: Number of agents to generate
        num_rounds: Number of simulation rounds
        interactions_per_round: Average interactions per round
    
    Returns:
        Dictionary with cost estimates
    """
    # Approximate token usage per operation
    TOKENS_PER_AGENT_GEN = 3000  # Biography, personality, etc.
    TOKENS_PER_CONVERSATION = 2000  # Typical conversation
    TOKENS_PER_ANALYSIS = 500  # Post-conversation analysis
    
    # Pricing per 1K tokens (Updated December 2024)
    PRICING = {
        # OpenAI models - prices per 1K tokens
        "openai:gpt-4o": {"input": 0.003, "output": 0.01},  # $3/$10 per 1M tokens
        "openai:gpt-4o-2024-11-20": {"input": 0.003, "output": 0.01},
        "openai:gpt-4o-mini": {"input": 0.00015, "output": 0.0006},  # $0.15/$0.60 per 1M tokens
        "openai:gpt-4-turbo": {"input": 0.01, "output": 0.03},  # $10/$30 per 1M tokens
        "openai:gpt-4-turbo-2024-04-09": {"input": 0.01, "output": 0.03},
        "openai:gpt-4": {"input": 0.03, "output": 0.06},  # $30/$60 per 1M tokens
        "openai:gpt-4-32k": {"input": 0.06, "output": 0.12},  # $60/$120 per 1M tokens
        "openai:gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},  # $0.50/$1.50 per 1M tokens
        "openai:gpt-3.5-turbo-0125": {"input": 0.0005, "output": 0.0015},
        
        # Anthropic models - prices per 1K tokens
        "anthropic:claude-opus-4": {"input": 0.015, "output": 0.075},  # $15/$75 per 1M tokens
        "anthropic:claude-sonnet-4": {"input": 0.003, "output": 0.015},  # $3/$15 per 1M tokens
        "anthropic:claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
        "anthropic:claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
        "anthropic:claude-3-5-haiku": {"input": 0.0008, "output": 0.004},  # $0.80/$4 per 1M tokens
        
        # Previous generation Anthropic models
        "anthropic:claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
        "anthropic:claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
        "anthropic:claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
    }
    
    if model_spec not in PRICING:
        return {
            "warning": "Pricing not available for this model",
            "total_tokens": 0,
            "estimated_cost": 0
        }
    
    # Calculate total tokens
    agent_gen_tokens = num_agents * TOKENS_PER_AGENT_GEN
    conversation_tokens = num_rounds * interactions_per_round * TOKENS_PER_CONVERSATION
    analysis_tokens = num_rounds * interactions_per_round * TOKENS_PER_ANALYSIS
    
    total_tokens = agent_gen_tokens + conversation_tokens + analysis_tokens
    
    # Estimate cost (assuming 50/50 input/output split)
    price = PRICING[model_spec]
    input_cost = (total_tokens * 0.5 / 1000) * price["input"]
    output_cost = (total_tokens * 0.5 / 1000) * price["output"]
    total_cost = input_cost + output_cost
    
    return {
        "agent_generation_tokens": agent_gen_tokens,
        "conversation_tokens": conversation_tokens,
        "analysis_tokens": analysis_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(total_cost, 2),
        "breakdown": {
            "agent_generation": round(agent_gen_tokens / 1000 * 
                                    (price["input"] + price["output"]) / 2, 2),
            "conversations": round(conversation_tokens / 1000 * 
                                 (price["input"] + price["output"]) / 2, 2),
            "analysis": round(analysis_tokens / 1000 * 
                            (price["input"] + price["output"]) / 2, 2)
        }
    }


def recommend_model(budget: float, num_agents: int, 
                   num_rounds: int, quality_priority: float = 0.5) -> str:
    """Recommend a model based on budget and requirements.
    
    Args:
        budget: Maximum budget in USD
        num_agents: Number of agents
        num_rounds: Number of simulation rounds  
        quality_priority: 0-1 scale, where 0 prioritizes cost and 1 prioritizes quality
    
    Returns:
        Recommended model specification
    """
    models = [
        # OpenAI models
        ("openai:gpt-4", 1.0, "Original GPT-4 - Highest quality, highest cost"),
        ("openai:gpt-4-turbo", 0.95, "GPT-4 Turbo - High quality, 128K context, better cost"),
        ("openai:gpt-4o", 0.9, "GPT-4o - Multimodal, fast, very cost-effective"),
        ("openai:gpt-3.5-turbo", 0.6, "GPT-3.5 Turbo - Good quality, very low cost"),
        ("openai:gpt-4o-mini", 0.5, "GPT-4o Mini - Basic quality, minimal cost"),
        
        # Anthropic models
        ("anthropic:claude-opus-4", 0.98, "Claude Opus 4 - Most capable, expensive"),
        ("anthropic:claude-sonnet-4", 0.92, "Claude Sonnet 4 - Excellent balance"),
        ("anthropic:claude-3-5-sonnet", 0.91, "Claude 3.5 Sonnet - Great performance"),
        ("anthropic:claude-3-5-haiku", 0.7, "Claude 3.5 Haiku - Fast and affordable"),
    ]
    
    viable_models = []
    
    for model, quality_score, desc in models:
        cost_est = estimate_cost(model, num_agents, num_rounds, 
                               interactions_per_round=num_agents//2)
        
        if cost_est.get("estimated_cost_usd", float('inf')) <= budget:
            # Calculate combined score
            cost_score = 1 - (cost_est["estimated_cost_usd"] / budget)
            combined_score = (quality_priority * quality_score + 
                            (1 - quality_priority) * cost_score)
            viable_models.append((model, combined_score, desc, 
                                cost_est["estimated_cost_usd"]))
    
    if not viable_models:
        return "mock:test"  # Budget too low for any real model
    
    # Sort by combined score
    viable_models.sort(key=lambda x: x[1], reverse=True)
    
    best_model, score, desc, cost = viable_models[0]
    print(f"Recommended: {best_model} - {desc}")
    print(f"Estimated cost: ${cost:.2f} (within budget of ${budget:.2f})")
    
    return best_model