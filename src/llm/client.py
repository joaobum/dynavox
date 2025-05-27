"""LLM client interfaces for different providers."""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import json
import os
import logging

logger = logging.getLogger('dynavox.llm')


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.7, 
                max_tokens: int = 1000) -> str:
        """Generate text from a prompt."""
        pass
    
    @abstractmethod
    def generate_json(self, prompt: str, temperature: float = 0.7, 
                     max_tokens: int = 1000) -> Dict[str, Any]:
        """Generate and parse JSON response."""
        pass


class OpenAIClient(LLMClient):
    """OpenAI API client implementation."""
    
    # Available OpenAI models (Updated December 2024)
    AVAILABLE_MODELS = [
        # Latest models
        "gpt-4o",
        "gpt-4o-2024-11-20",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4-turbo-2024-04-09",
        "gpt-4-turbo-preview",
        "gpt-4",
        "gpt-4-32k",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo-16k",
        # Legacy model names still supported
        "gpt-4-1106-preview",
        "gpt-4-0125-preview",
    ]
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        try:
            from openai import OpenAI
        except ImportError:
            logger.error("OpenAI package not installed")
            raise ImportError("Please install openai: pip install openai")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.error("OpenAI API key not provided")
            raise ValueError("OpenAI API key not provided")
        
        # Validate model selection
        if model not in self.AVAILABLE_MODELS:
            logger.warning(f"Model '{model}' not in standard list of available models")
            print(f"Warning: Model '{model}' not in standard list. Available models:")
            for m in self.AVAILABLE_MODELS:
                print(f"  - {m}")
            print(f"Proceeding with '{model}' anyway...")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        logger.info(f"Initialized OpenAI client with model: {self.model}")
        print(f"Initialized OpenAI client with model: {self.model}")
    
    def generate(self, prompt: str, temperature: float = 0.7, 
                max_tokens: int = 1000) -> str:
        """Generate text using OpenAI API."""
        logger.debug(f"Generating response with model {self.model}, temp={temperature}, max_tokens={max_tokens}")
        logger.debug(f"LLM Prompt:\n{prompt}")
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        result = response.choices[0].message.content
        logger.debug(f"LLM Response:\n{result}")
        return result
    
    def generate_json(self, prompt: str, temperature: float = 0.7, 
                     max_tokens: int = 1000) -> Dict[str, Any]:
        """Generate JSON response using OpenAI API."""
        logger.debug("Generating JSON response")
        
        # Add JSON instruction to prompt if not present
        if "JSON" not in prompt:
            prompt += "\n\nRespond with valid JSON only."
        
        response_text = self.generate(prompt, temperature, max_tokens)
        
        # Extract JSON from response
        try:
            # Try to find JSON in the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                result = json.loads(json_str)
                logger.debug(f"Successfully parsed JSON with {len(result)} keys")
                return result
            else:
                logger.error("No JSON found in response")
                raise ValueError("No JSON found in response")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            raise ValueError(f"Failed to parse JSON response: {e}\nResponse: {response_text}")


class AnthropicClient(LLMClient):
    """Anthropic Claude API client implementation."""
    
    # Available Anthropic models (Updated December 2024)
    AVAILABLE_MODELS = [
        # Latest models
        "claude-opus-4",
        "claude-sonnet-4",
        "claude-3-5-sonnet",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku",
        # Previous generation
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        # Legacy naming still supported
        "claude-3-opus",
        "claude-3-sonnet",
        "claude-3-haiku",
    ]
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet"):
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")
        
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key not provided")
        
        # Validate model selection
        if model not in self.AVAILABLE_MODELS:
            print(f"Warning: Model '{model}' not in standard list. Available models:")
            for m in self.AVAILABLE_MODELS:
                print(f"  - {m}")
            print(f"Proceeding with '{model}' anyway...")
        
        self.client = Anthropic(api_key=self.api_key)
        self.model = model
        
        # Map common model names to actual API model IDs
        model_mapping = {
            "claude-3-5-haiku": "claude-3-haiku-20240307",
            "claude-3-haiku": "claude-3-haiku-20240307",
            "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
            "claude-3-sonnet": "claude-3-sonnet-20240229",
            "claude-3-opus": "claude-3-opus-20240229",
        }
        
        # Use mapped model name if available
        if self.model in model_mapping:
            actual_model = model_mapping[self.model]
            print(f"Mapping {self.model} to {actual_model}")
            self.model = actual_model
        
        print(f"Initialized Anthropic client with model: {self.model}")
    
    def generate(self, prompt: str, temperature: float = 0.7, 
                max_tokens: int = 1000) -> str:
        """Generate text using Anthropic API."""
        logger.debug(f"Generating response with model {self.model}, temp={temperature}, max_tokens={max_tokens}")
        logger.debug(f"LLM Prompt:\n{prompt}")
        
        try:
            response = self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            result = response.content[0].text
            logger.debug(f"LLM Response:\n{result}")
            return result
        except Exception as e:
            if "authentication_error" in str(e):
                raise ValueError(
                    "Anthropic API authentication failed. Please check:\n"
                    "1. Your ANTHROPIC_API_KEY in the .env file is valid\n"
                    "2. The API key has not expired\n"
                    "3. You have access to the requested model\n"
                    f"Original error: {e}"
                )
            raise
    
    def generate_json(self, prompt: str, temperature: float = 0.7, 
                     max_tokens: int = 1000) -> Dict[str, Any]:
        """Generate JSON response using Anthropic API."""
        logger.debug("Generating JSON response")
        
        # Add JSON instruction to prompt if not present
        if "JSON" not in prompt:
            prompt += "\n\nRespond with valid JSON only."
        
        response_text = self.generate(prompt, temperature, max_tokens)
        
        # Extract JSON from response
        try:
            # Try to find JSON in the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                result = json.loads(json_str)
                logger.debug(f"Successfully parsed JSON with {len(result)} keys")
                return result
            else:
                logger.error("No JSON found in response")
                raise ValueError("No JSON found in response")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            raise ValueError(f"Failed to parse JSON response: {e}\nResponse: {response_text}")


class MockLLMClient(LLMClient):
    """Mock LLM client for testing without API calls."""
    
    def __init__(self):
        self.call_count = 0
        self.delay = 0.01  # Minimal delay for faster mock testing
        logger.info("Initialized MockLLMClient for testing")
    
    def generate(self, prompt: str, temperature: float = 0.7, 
                max_tokens: int = 1000) -> str:
        """Generate mock response."""
        import time
        time.sleep(self.delay)  # Simulate API latency
        self.call_count += 1
        logger.debug(f"Mock LLM Prompt:\n{prompt}")
        
        # Generate more natural mock conversations
        if "conversation" in prompt.lower() and "your turn to speak" in prompt.lower():
            # Natural conversation responses
            responses = [
                "I see your point, but have you considered the economic impact?",
                "That's exactly what I've been thinking lately.",
                "Hmm, I'm not so sure about that.",
                "The way I see it, we need to be more practical here.",
                "Sure, but what about the people who'll be affected?",
                "Look, I get where you're coming from, but...",
                "Yeah, that makes sense actually.",
                "I don't know, seems like we're missing something.",
                "From my experience, it's not that simple.",
                "Right, but how do we actually implement that?",
                "That's fair. I hadn't thought of it that way.",
                "Well, depends on how you look at it.",
                "Interesting perspective. Where'd you hear about that?",
                "I gotta go soon, but this has been interesting.",
                "Nice talking with you. Take care!",
                "Anyway, I should get going.",
                "Good chat! See you around."
            ]
            
            # Return ending phrases more often after several turns
            if self.call_count > 10 and self.call_count % 3 == 0:
                return responses[-4:][self.call_count % 4]
            
            result = responses[self.call_count % (len(responses) - 4)]
        else:
            result = f"Mock response #{self.call_count} to prompt: {prompt[:50]}..."
        
        logger.debug(f"Mock LLM Response:\\n{result}")
        return result
    
    def generate_json(self, prompt: str, temperature: float = 0.7, 
                     max_tokens: int = 1000) -> Dict[str, Any]:
        """Generate mock JSON response."""
        import time
        time.sleep(self.delay)  # Simulate API latency
        self.call_count += 1
        
        # Return different mock data based on prompt content
        if "personality" in prompt.lower() and "hexaco" in prompt.lower():
            # Add variation based on call count to avoid singular covariance matrix
            return {
                "honesty_humility": 40 + (self.call_count * 7) % 50,
                "emotionality": 30 + (self.call_count * 11) % 60,
                "extraversion": 45 + (self.call_count * 13) % 40,
                "agreeableness": 50 + (self.call_count * 5) % 45,
                "conscientiousness": 35 + (self.call_count * 17) % 55,
                "openness": 40 + (self.call_count * 19) % 50
            }
        elif "background" in prompt.lower() and "demographic" in prompt.lower():
            # Diverse occupations for mock mode
            occupations = [
                "Teacher", "Software Engineer", "Nurse", "Marketing Manager",
                "Electrician", "Restaurant Owner", "Social Worker", "Lawyer",
                "Graphic Designer", "Sales Representative", "Mechanic", "Professor",
                "Real Estate Agent", "Chef", "Accountant", "Police Officer",
                "Writer", "Pharmacist", "Construction Worker", "HR Manager"
            ]
            occupation = occupations[self.call_count % len(occupations)]
            
            # Match education to occupation
            education_map = {
                "Teacher": ("masters", "Education"),
                "Software Engineer": ("bachelors", "Computer Science"),
                "Nurse": ("bachelors", "Nursing"),
                "Professor": ("phd", "History"),
                "Lawyer": ("masters", "Law"),
                "Mechanic": ("high_school", "Vocational Training"),
                "Construction Worker": ("high_school", "Trade School"),
                "Chef": ("high_school", "Culinary Arts")
            }
            
            edu_level, edu_field = education_map.get(occupation, ("bachelors", "Business"))
            
            # Vary socioeconomic tags
            socio_tags = [
                ["middle-class", "urban"],
                ["working-class", "suburban"],
                ["upper-middle-class", "urban"],
                ["middle-class", "rural"],
                ["working-class", "urban"]
            ]
            
            return {
                "name": f"Mock Person {self.call_count}",
                "age": 30 + (self.call_count % 40),
                "occupation": occupation,
                "education_level": edu_level,
                "education_field": edu_field,
                "socioeconomic_tags": socio_tags[self.call_count % len(socio_tags)],
                "relationship_tags": ["married", "parent-of-one"] if self.call_count % 2 == 0 else ["single"],
                "cultural_tags": ["progressive", "secular"] if self.call_count % 3 != 0 else ["moderate", "traditional"]
            }
        elif "emotional" in prompt.lower() and "baseline" in prompt.lower():
            return {
                "dispositional_affect": -20 + (self.call_count * 23) % 40,
                "stress_tolerance": 40 + (self.call_count * 7) % 50,
                "social_confidence": 45 + (self.call_count * 11) % 45,
                "self_efficacy": 50 + (self.call_count * 13) % 40
            }
        elif "opinion" in prompt.lower() and "topic" in prompt.lower():
            # Vary opinions slightly based on call count
            base_position = (self.call_count * 17) % 200 - 100
            return {
                "position": base_position,
                "certainty": 60 + (self.call_count % 30),
                "importance": 50 + (self.call_count % 40),
                "knowledge": 60 + (self.call_count % 30),
                "emotional_charge": 30 + (self.call_count % 50)
            }
        elif "analyze" in prompt.lower() and "conversation" in prompt.lower():
            return {
                "topics_discussed": ["climate_change", "wealth_inequality"],
                "agent1_perspective": {
                    "arguments_made": ["We need action now", "The science is clear"],
                    "arguments_encountered": ["Economic concerns", "Implementation challenges"],
                    "interaction_quality": 0.7,
                    "validation_received": 0.4,
                    "conflict_level": 0.3
                },
                "agent2_perspective": {
                    "arguments_made": ["Economic concerns", "Implementation challenges"],
                    "arguments_encountered": ["We need action now", "The science is clear"],
                    "interaction_quality": 0.7,
                    "validation_received": 0.5,
                    "conflict_level": 0.3
                }
            }
        else:
            return {"response": f"Mock JSON response #{self.call_count}"}