#!/usr/bin/env python3
"""
Test Anthropic models to find correct model names
"""
import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

def test_anthropic_models():
    """Test different model names with Anthropic API."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("No ANTHROPIC_API_KEY found in environment")
        return
    
    client = Anthropic(api_key=api_key)
    
    # Model names to test
    model_names = [
        "claude-3-5-haiku",
        "claude-3.5-haiku", 
        "claude-3-5-haiku-20241022",
        "claude-3-haiku-20240307",
        "claude-3-haiku",
        "claude-haiku",
        "claude-3-5-sonnet-20241022",  # This one should work
    ]
    
    test_prompt = "Say 'Hello' in one word."
    
    for model in model_names:
        print(f"\nTesting model: {model}")
        try:
            response = client.messages.create(
                model=model,
                max_tokens=10,
                messages=[{"role": "user", "content": test_prompt}]
            )
            print(f"✅ Success! Response: {response.content[0].text}")
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    print("\n\nTesting with known working model (claude-3-5-sonnet-20241022):")
    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=10,
            messages=[{"role": "user", "content": test_prompt}]
        )
        print(f"✅ Success! Response: {response.content[0].text}")
    except Exception as e:
        print(f"❌ Failed: {e}")

if __name__ == "__main__":
    test_anthropic_models()