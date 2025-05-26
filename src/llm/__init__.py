"""LLM interface module for social dynamics modeling framework."""
from .client import LLMClient, OpenAIClient, AnthropicClient, MockLLMClient
from .prompts import PromptTemplates
from .model_selection import (
    get_available_models,
    create_llm_client,
    estimate_cost,
    recommend_model
)

__all__ = [
    'LLMClient', 
    'OpenAIClient', 
    'AnthropicClient',
    'MockLLMClient',
    'PromptTemplates',
    'get_available_models',
    'create_llm_client',
    'estimate_cost',
    'recommend_model'
]