"""Async LLM client interfaces for parallel API calls."""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import json
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor


class AsyncLLMClient(ABC):
    """Abstract base class for async LLM clients."""
    
    @abstractmethod
    async def generate(self, prompt: str, temperature: float = 0.7, 
                      max_tokens: int = 1000) -> str:
        """Generate text from a prompt asynchronously."""
        pass
    
    @abstractmethod
    async def generate_json(self, prompt: str, temperature: float = 0.7, 
                           max_tokens: int = 1000) -> Dict[str, Any]:
        """Generate and parse JSON response asynchronously."""
        pass
    
    @abstractmethod
    async def generate_batch(self, prompts: List[str], temperature: float = 0.7,
                           max_tokens: int = 1000) -> List[str]:
        """Generate multiple responses in parallel."""
        pass


class AsyncOpenAIClient(AsyncLLMClient):
    """Async OpenAI API client implementation."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4", 
                 max_concurrent: int = 10):
        """Initialize async OpenAI client.
        
        Args:
            api_key: OpenAI API key
            model: Model to use
            max_concurrent: Maximum concurrent API calls
        """
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrent)
        print(f"Initialized Async OpenAI client with model: {self.model}")
    
    async def generate(self, prompt: str, temperature: float = 0.7, 
                      max_tokens: int = 1000) -> str:
        """Generate text using OpenAI API asynchronously."""
        async with self.semaphore:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
    
    async def generate_json(self, prompt: str, temperature: float = 0.7, 
                           max_tokens: int = 1000) -> Dict[str, Any]:
        """Generate JSON response using OpenAI API asynchronously."""
        if "JSON" not in prompt:
            prompt += "\n\nRespond with valid JSON only."
        
        response_text = await self.generate(prompt, temperature, max_tokens)
        
        try:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}\nResponse: {response_text}")
    
    async def generate_batch(self, prompts: List[str], temperature: float = 0.7,
                           max_tokens: int = 1000) -> List[str]:
        """Generate multiple responses in parallel."""
        tasks = [self.generate(prompt, temperature, max_tokens) for prompt in prompts]
        return await asyncio.gather(*tasks)


class AsyncLLMWrapper:
    """Wrapper to use sync LLM clients in async context."""
    
    def __init__(self, sync_client, max_workers: int = 10):
        """Wrap a synchronous LLM client for async usage.
        
        Args:
            sync_client: Synchronous LLM client instance
            max_workers: Maximum thread workers for parallel execution
        """
        self.sync_client = sync_client
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def generate(self, prompt: str, temperature: float = 0.7, 
                      max_tokens: int = 1000) -> str:
        """Generate text asynchronously using thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.sync_client.generate,
            prompt, temperature, max_tokens
        )
    
    async def generate_json(self, prompt: str, temperature: float = 0.7, 
                           max_tokens: int = 1000) -> Dict[str, Any]:
        """Generate JSON asynchronously using thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.sync_client.generate_json,
            prompt, temperature, max_tokens
        )
    
    async def generate_batch(self, prompts: List[str], temperature: float = 0.7,
                           max_tokens: int = 1000) -> List[str]:
        """Generate multiple responses in parallel using thread pool."""
        tasks = [self.generate(prompt, temperature, max_tokens) for prompt in prompts]
        return await asyncio.gather(*tasks)
    
    def close(self):
        """Cleanup thread pool executor."""
        self.executor.shutdown(wait=True)


class AsyncMockLLMClient(AsyncLLMClient):
    """Mock async LLM client for testing."""
    
    def __init__(self, delay: float = 0.1):
        """Initialize mock client with optional delay.
        
        Args:
            delay: Simulated API call delay in seconds
        """
        self.call_count = 0
        self.delay = delay
        self.lock = asyncio.Lock()
    
    async def generate(self, prompt: str, temperature: float = 0.7, 
                      max_tokens: int = 1000) -> str:
        """Generate mock response asynchronously."""
        await asyncio.sleep(self.delay)  # Simulate API delay
        
        async with self.lock:
            self.call_count += 1
            call_num = self.call_count
        
        return f"Mock async response #{call_num} to prompt: {prompt[:50]}..."
    
    async def generate_json(self, prompt: str, temperature: float = 0.7, 
                           max_tokens: int = 1000) -> Dict[str, Any]:
        """Generate mock JSON response asynchronously."""
        await asyncio.sleep(self.delay)
        
        async with self.lock:
            self.call_count += 1
            call_num = self.call_count
        
        # Return appropriate mock data based on prompt
        if "personality" in prompt.lower() and "hexaco" in prompt.lower():
            return {
                "honesty_humility": 40 + (call_num * 7) % 50,
                "emotionality": 30 + (call_num * 11) % 60,
                "extraversion": 45 + (call_num * 13) % 40,
                "agreeableness": 50 + (call_num * 5) % 45,
                "conscientiousness": 35 + (call_num * 17) % 55,
                "openness": 40 + (call_num * 19) % 50
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
            return {"response": f"Mock async JSON response #{call_num}"}
    
    async def generate_batch(self, prompts: List[str], temperature: float = 0.7,
                           max_tokens: int = 1000) -> List[str]:
        """Generate multiple mock responses in parallel."""
        tasks = [self.generate(prompt, temperature, max_tokens) for prompt in prompts]
        return await asyncio.gather(*tasks)