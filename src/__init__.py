"""OpinionDynamics: Agent-Based Social Dynamics Modeling Framework."""

from .agents import Agent, PersonalityTraits, Background, Opinion
from .simulation import SimulationEngine
from .llm import LLMClient, OpenAIClient, AnthropicClient, create_llm_client
from .analysis import SimulationVisualizer, SimulationReporter
from .quick_sim import QuickSimulation

__version__ = "0.1.0"

__all__ = [
    'Agent',
    'PersonalityTraits',
    'Background',
    'Opinion',
    'SimulationEngine',
    'LLMClient',
    'OpenAIClient',
    'AnthropicClient',
    'create_llm_client',
    'SimulationVisualizer',
    'SimulationReporter',
    'QuickSimulation'
]