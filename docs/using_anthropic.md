# Using Anthropic Claude Models with DynaVox

## Available Models

DynaVox supports all current Anthropic Claude models:

- **Claude 3.5 Haiku** (`claude-3-5-haiku`) - Fast and affordable
- **Claude 3.5 Sonnet** (`claude-3-5-sonnet`) - Balanced performance  
- **Claude Opus 4** (`claude-opus-4`) - Most powerful but expensive
- **Claude Sonnet 4** (`claude-sonnet-4`) - Good balance of quality and cost

## Setup

1. **Get an API Key**
   - Sign up at https://console.anthropic.com
   - Create an API key in your account settings
   
2. **Add to .env file**
   ```
   ANTHROPIC_API_KEY=your-api-key-here
   ```

3. **Verify your key is valid**
   ```bash
   python examples/test_anthropic_models.py
   ```

## Running with Claude Models

### Command Line Examples

```bash
# Test with Claude 3.5 Haiku (fast and affordable)
python examples/diverse_agents_example.py --real --model claude-3-5-haiku

# Use Claude 3.5 Sonnet with async
python examples/diverse_agents_example.py --real --async --model claude-3-5-sonnet

# Use Claude Opus 4 for highest quality
python examples/diverse_agents_example.py --real --model claude-opus-4
```

### Python Code Examples

```python
from src import QuickSimulation

# Using Claude 3.5 Haiku
sim = QuickSimulation(model='claude-3-5-haiku')

# Using Claude 3.5 Sonnet with async
sim = QuickSimulation(model='claude-3-5-sonnet', use_async=True)
```

## Model Selection Guide

| Model | Best For | Cost | Speed |
|-------|----------|------|-------|
| claude-3-5-haiku | Testing, large simulations | $ | Fast |
| claude-3-5-sonnet | Production, balanced needs | $$ | Medium |
| claude-opus-4 | Research, highest quality | $$$$ | Slower |

## Cost Estimates

For a simulation with 20 agents and 15 rounds:
- **Claude 3.5 Haiku**: ~$2-5
- **Claude 3.5 Sonnet**: ~$10-20  
- **Claude Opus 4**: ~$50-100

## Troubleshooting

### Authentication Error
If you see "invalid x-api-key":
1. Check your API key is correctly set in .env
2. Verify the key hasn't expired
3. Ensure you have access to the requested model
4. Try regenerating your API key

### Model Not Found
The framework automatically maps common model names to their API versions:
- `claude-3-5-haiku` → `claude-3-haiku-20240307`
- `claude-3-5-sonnet` → `claude-3-5-sonnet-20241022`

### Rate Limits
Anthropic has rate limits. If you hit them:
- Use async mode to better manage concurrent requests
- Reduce the number of agents or rounds
- Add delays between rounds in your simulation