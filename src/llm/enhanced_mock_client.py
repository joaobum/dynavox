"""Enhanced mock LLM client that generates varied conversation types."""
from typing import Dict, Any
import random
import json
from .client import MockLLMClient


class EnhancedMockLLMClient(MockLLMClient):
    """Enhanced mock client that generates realistic conversation dynamics."""
    
    def __init__(self):
        super().__init__()
        self.conversation_scenario = 0
        
    def generate_json(self, prompt: str, temperature: float = 0.7, 
                     max_tokens: int = 1000) -> Dict[str, Any]:
        """Generate mock JSON with varied conversation types."""
        
        # Use parent's implementation for most cases
        if "analyze" not in prompt.lower() or "conversation" not in prompt.lower():
            return super().generate_json(prompt, temperature, max_tokens)
        
        # Generate varied conversation analyses
        self.conversation_scenario = (self.conversation_scenario + 1) % 5
        
        if self.conversation_scenario == 0:
            # Echo chamber scenario
            return {
                "topics_discussed": ["climate_change", "wealth_inequality"],
                "agent1_perspective": {
                    "arguments_made": ["We absolutely need urgent action", "I completely agree with your points"],
                    "arguments_encountered": ["Yes, the evidence is overwhelming", "Exactly what I was thinking"],
                    "interaction_quality": 0.85,
                    "validation_received": 0.9,
                    "conflict_level": 0.1
                },
                "agent2_perspective": {
                    "arguments_made": ["Yes, the evidence is overwhelming", "Exactly what I was thinking"],
                    "arguments_encountered": ["We absolutely need urgent action", "I completely agree with your points"],
                    "interaction_quality": 0.85,
                    "validation_received": 0.9,
                    "conflict_level": 0.1
                }
            }
        
        elif self.conversation_scenario == 1:
            # Heated debate scenario
            return {
                "topics_discussed": ["climate_change", "wealth_inequality"],
                "agent1_perspective": {
                    "arguments_made": ["You're completely wrong about this", "That's just not realistic"],
                    "arguments_encountered": ["No, you don't understand the economics", "Your ideas would destroy jobs"],
                    "interaction_quality": 0.3,
                    "validation_received": 0.1,
                    "conflict_level": 0.9
                },
                "agent2_perspective": {
                    "arguments_made": ["No, you don't understand the economics", "Your ideas would destroy jobs"],
                    "arguments_encountered": ["You're completely wrong about this", "That's just not realistic"],
                    "interaction_quality": 0.3,
                    "validation_received": 0.1,
                    "conflict_level": 0.9
                }
            }
        
        elif self.conversation_scenario == 2:
            # Respectful disagreement scenario
            return {
                "topics_discussed": ["climate_change", "wealth_inequality"],
                "agent1_perspective": {
                    "arguments_made": ["I see your point, but consider this perspective", "That's valid, though I have concerns about implementation"],
                    "arguments_encountered": ["I understand where you're coming from, but what about", "Fair point, but we also need to think about"],
                    "interaction_quality": 0.75,
                    "validation_received": 0.6,
                    "conflict_level": 0.5
                },
                "agent2_perspective": {
                    "arguments_made": ["I understand where you're coming from, but what about", "Fair point, but we also need to think about"],
                    "arguments_encountered": ["I see your point, but consider this perspective", "That's valid, though I have concerns about implementation"],
                    "interaction_quality": 0.75,
                    "validation_received": 0.6,
                    "conflict_level": 0.5
                }
            }
        
        elif self.conversation_scenario == 3:
            # Productive exchange scenario
            return {
                "topics_discussed": ["climate_change", "wealth_inequality"],
                "agent1_perspective": {
                    "arguments_made": ["Here's what the research shows", "I've seen this work in practice"],
                    "arguments_encountered": ["That's a really good point", "I hadn't considered that angle"],
                    "interaction_quality": 0.8,
                    "validation_received": 0.7,
                    "conflict_level": 0.2
                },
                "agent2_perspective": {
                    "arguments_made": ["That's a really good point", "I hadn't considered that angle"],
                    "arguments_encountered": ["Here's what the research shows", "I've seen this work in practice"],
                    "interaction_quality": 0.8,
                    "validation_received": 0.5,
                    "conflict_level": 0.2
                }
            }
        
        else:
            # Neutral conversation
            return {
                "topics_discussed": ["climate_change", "wealth_inequality"],
                "agent1_perspective": {
                    "arguments_made": ["I think this is important", "We should consider both sides"],
                    "arguments_encountered": ["That's one way to look at it", "There are pros and cons"],
                    "interaction_quality": 0.5,
                    "validation_received": 0.4,
                    "conflict_level": 0.3
                },
                "agent2_perspective": {
                    "arguments_made": ["That's one way to look at it", "There are pros and cons"],
                    "arguments_encountered": ["I think this is important", "We should consider both sides"],
                    "interaction_quality": 0.5,
                    "validation_received": 0.4,
                    "conflict_level": 0.3
                }
            }