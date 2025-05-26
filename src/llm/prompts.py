"""Prompt templates for LLM-based agent generation and interaction."""


class PromptTemplates:
    """Collection of prompt templates for various LLM tasks."""
    
    GENERATE_PERSONALITY = """Generate a psychologically realistic personality profile using the HEXACO model.
        
Each trait should be a value from 0-100 where:
- 0-20: Very low
- 21-40: Low  
- 41-60: Average
- 61-80: High
- 81-100: Very high

The traits are:
1. Honesty-Humility: Sincerity, fairness, greed avoidance, modesty
2. Emotionality: Fearfulness, anxiety, dependence, sentimentality  
3. Extraversion: Social self-esteem, social boldness, sociability, liveliness
4. Agreeableness: Forgiveness, gentleness, flexibility, patience
5. Conscientiousness: Organization, diligence, perfectionism, prudence
6. Openness: Aesthetic appreciation, inquisitiveness, creativity, unconventionality

Generate a coherent personality avoiding impossible combinations (e.g., very high extraversion with very high anxiety is uncommon).

{constraints}

Respond with ONLY a JSON object in this exact format:
{{
  "honesty_humility": <int>,
  "emotionality": <int>,
  "extraversion": <int>,
  "agreeableness": <int>,
  "conscientiousness": <int>,
  "openness": <int>
}}"""

    GENERATE_BACKGROUND = """Given the following personality profile, generate a realistic demographic background.

Personality Profile:
- Honesty-Humility: {honesty_humility}/100
- Emotionality: {emotionality}/100  
- Extraversion: {extraversion}/100
- Agreeableness: {agreeableness}/100
- Conscientiousness: {conscientiousness}/100
- Openness: {openness}/100

Create a background that aligns with this personality. Consider diverse life paths including:
- Unemployment, service jobs, manual labor, skilled trades
- Various education levels from no high school to PhD
- Different socioeconomic backgrounds from poverty to wealthy

Remember:
- Not everyone has traditional career paths
- Education doesn't always match occupation
- Consider realistic barriers and opportunities

Include:
1. Full name (realistic for the demographic)
2. Age (18-80)
3. Occupation (specific role and seniority level)
4. Education level and field
5. 2-4 socioeconomic tags from: poverty, working-poor, working-class, middle-class, upper-middle-class, wealthy, rural, urban, suburban, first-generation, immigrant-family, public-assistance, paycheck-to-paycheck, debt-burdened, homeowner, renter, homeless
6. 2-3 relationship tags (e.g., "single", "married", "divorced", "widowed", "parent-of-two", "childless", "caregiver", "estranged-family")
7. 2-3 cultural tags (e.g., "progressive", "traditional", "religious", "secular", "community-oriented", "individualistic")

{constraints}

Respond with ONLY a JSON object matching this structure:
{{
  "name": "<full name>",
  "age": <int>,
  "occupation": "<specific role>",
  "education_level": "<no_high_school|high_school|some_college|associates|bachelors|masters|phd>",
  "education_field": "<field of study>",
  "socioeconomic_tags": ["tag1", "tag2", ...],
  "relationship_tags": ["tag1", "tag2", ...],
  "cultural_tags": ["tag1", "tag2", ...]
}}"""

    GENERATE_EMOTIONAL_BASELINE = """Based on the personality and background below, determine stable emotional tendencies.

PERSONALITY:
- Honesty-Humility: {honesty_humility}/100
- Emotionality: {emotionality}/100
- Extraversion: {extraversion}/100
- Agreeableness: {agreeableness}/100
- Conscientiousness: {conscientiousness}/100
- Openness: {openness}/100

BACKGROUND:
- Age: {age}
- Occupation: {occupation}
- Relationship status: {relationship_status}
- Socioeconomic: {socioeconomic}

Determine:
1. Dispositional affect (-50 to +50): General tendency toward positive or negative mood
2. Stress tolerance (0-100): Ability to handle stressors without becoming overwhelmed
3. Social confidence (0-100): Comfort and ease in social situations
4. Self-efficacy (0-100): Belief in ability to achieve goals and handle challenges

Consider how personality traits interact with life circumstances to create these baselines.

Respond with ONLY a JSON object:
{{
  "dispositional_affect": <int between -50 and 50>,
  "stress_tolerance": <int between 0 and 100>,
  "social_confidence": <int between 0 and 100>,
  "self_efficacy": <int between 0 and 100>
}}"""

    GENERATE_BIOGRAPHY = """Write a biographical narrative for the following person.

PROFILE:
Name: {name}
Age: {age}
Occupation: {occupation}
Education: {education_level} in {education_field}

PERSONALITY (HEXACO):
- Honesty-Humility: {honesty_humility}/100 
- Emotionality: {emotionality}/100
- Extraversion: {extraversion}/100
- Agreeableness: {agreeableness}/100
- Conscientiousness: {conscientiousness}/100
- Openness: {openness}/100

BACKGROUND CONTEXT:
- Socioeconomic: {socioeconomic}
- Relationships: {relationships}
- Cultural: {cultural}

EMOTIONAL TENDENCIES:
- General mood: {mood_tendency}
- Stress tolerance: {stress_tolerance}/100
- Social confidence: {social_confidence}/100
- Self-efficacy: {self_efficacy}/100

Write a 400-500 word biography that:
1. Explains key formative experiences that shaped their personality
2. Details their career path and why it suits them
3. Describes their core values and worldview
4. Explains their relationship patterns and social circles
5. Mentions 1-2 defining hobbies or interests

Focus on what makes them unique and authentic. Do NOT include conversation style or speech patterns - those will be handled separately."""

    GENERATE_CONVERSATION_STYLE = """Generate a detailed conversation style description for this person.

PROFILE:
Name: {name}
Age: {age}
Occupation: {occupation}
Education: {education_level} in {education_field}

PERSONALITY (HEXACO):
- Honesty-Humility: {honesty_humility}/100 
- Emotionality: {emotionality}/100
- Extraversion: {extraversion}/100
- Agreeableness: {agreeableness}/100
- Conscientiousness: {conscientiousness}/100
- Openness: {openness}/100

BACKGROUND:
- Socioeconomic: {socioeconomic}
- Cultural: {cultural}

EMOTIONAL BASELINE:
- Social confidence: {social_confidence}/100
- Self-efficacy: {self_efficacy}/100

Generate a 200-300 word description of their conversation style that includes:
1. Speech patterns and vocabulary level based on education/occupation
2. Typical phrases or expressions they use
3. How they structure their thoughts (linear vs. tangential)
4. Response length tendencies (brief vs. elaborate)
5. Use of humor, metaphors, or technical language
6. How their personality traits show up in conversation
7. Verbal habits when stressed, excited, or uncertain
8. Listening style and how they acknowledge others

Be specific and realistic. This should read like a guide for how to write dialogue for this character."""

    INITIALIZE_OPINION = """Based on this person's profile, determine their opinion on {topic}.

PROFILE SUMMARY:
Name: {name}
Occupation: {occupation}
Education: {education_level} in {education_field}
Cultural background: {cultural}

PERSONALITY HIGHLIGHTS:
- Openness: {openness}/100 (affects receptivity to change)
- Conscientiousness: {conscientiousness}/100 (affects need for evidence)
- Agreeableness: {agreeableness}/100 (affects concern for others)

BIOGRAPHY EXCERPT:
{biography_excerpt}

For the topic "{topic}", determine:
1. Position (-100 to +100): Their stance on the issue
2. Certainty (0-100): How sure they are of their position
3. Importance (0-100): How much this topic matters to them personally
4. Knowledge (0-100): How informed they believe themselves to be
5. Emotional charge (0-100): How emotionally invested they are

Consider how their background and personality would realistically shape their views.

Respond with ONLY a JSON object:
{{
  "position": <int between -100 and 100>,
  "certainty": <int between 0 and 100>,
  "importance": <int between 0 and 100>,
  "knowledge": <int between 0 and 100>,
  "emotional_charge": <int between 0 and 100>
}}"""

    CONVERSATION_PROMPT = """You are {name}, a {age}-year-old {occupation}.

BIOGRAPHY:
{biography}

YOUR PERSONALITY:
{personality_description}

CURRENT EMOTIONAL STATE:
- Energy: {arousal}/100
- Mood: {valence} (-50 to +50)
- Anxiety: {anxiety}/100
- Confidence: {confidence}/100

YOUR OPINIONS ON TODAY'S TOPICS:
{opinions_description}

CONVERSATION APPROACH:
Your intent is to {intent}. {intent_description}

BEHAVIORAL GUIDANCE:
{behavior_guide}

YOUR CONVERSATION STYLE:
{conversation_style}

CRITICAL INSTRUCTIONS FOR NATURAL CONVERSATION:
- Keep responses SHORT (1-3 sentences typical, max 4-5 only if absolutely necessary)
- NO GENERIC STARTERS: Avoid "Oh my gosh", "Oh wow", "Absolutely", "I totally agree" unless it genuinely fits YOUR specific personality and background
- Get to the point - don't acknowledge everything said
- Match your education/occupation:
  * Blue collar: "Look, I think...", "The way I see it..."
  * Academic: "The evidence suggests...", "In my experience..."
  * Business: "From a practical standpoint...", "The bottom line is..."
  * Creative: More colorful language, unique perspectives
  * Service: Focus on people and relationships
- Natural speech includes:
  * Contractions (I'm, don't, won't)
  * Occasional fillers ONLY if nervous/uncertain ("um", "well")
  * Incomplete thoughts when interrupted or changing mind
  * Different vocabulary levels based on education
- NEVER:
  * Explain your reasoning unless asked
  * List multiple points unless necessary
  * Use motivational speaker language
  * Sound like a chatbot or customer service rep

Remember: You're having a REAL conversation between two people who just met. Be natural, be brief, and most importantly - sound like a real person with your specific background, not a generic conversationalist. Your occupation, education, and life experiences should shine through in HOW you speak, not just WHAT you say."""

    ANALYZE_CONVERSATION = """Analyze this conversation between {agent1_name} and {agent2_name}.

CONVERSATION TRANSCRIPT:
{transcript}

AGENT 1 ({agent1_name}) PROFILE:
- Personality: {agent1_personality}
- Current intent: {agent1_intent}
- Key opinions: {agent1_opinions}

AGENT 2 ({agent2_name}) PROFILE:
- Personality: {agent2_personality}
- Current intent: {agent2_intent}
- Key opinions: {agent2_opinions}

Analyze:
1. Topics discussed and depth of engagement
2. Arguments presented by each agent
3. Interaction quality - Calculate based on:
   - RAPPORT (0-1): Measure of connection and mutual understanding
     * Active listening (acknowledging others' points)
     * Finding common ground
     * Respectful disagreement
     * Emotional attunement
   - CONFLICT (0-1): Level of tension or disagreement
     * Direct contradictions
     * Dismissive language
     * Emotional escalation
     * Personal attacks (if any)
   - UNDERSTANDING (0-1): Comprehension of each other's positions
     * Accurate paraphrasing
     * Asking clarifying questions
     * Building on others' ideas
     * Acknowledging complexity
   
   interaction_quality = (rapport + understanding + (1 - conflict)) / 3

4. Validation received (0-1):
   - Agreement with positions
   - Acknowledgment of good points
   - Emotional support
   - Respect for expertise

5. Potential influence based on:
   - Strength of arguments
   - Credibility established
   - Emotional resonance
   - Alignment with values

Respond with a JSON object:
{{
  "topics_discussed": ["topic1", "topic2", ...],
  "agent1_perspective": {{
    "arguments_made": ["arg1", "arg2", ...],
    "arguments_encountered": ["arg1", "arg2", ...],
    "interaction_quality": <float 0-1>,
    "validation_received": <float 0-1>,
    "conflict_level": <float 0-1>
  }},
  "agent2_perspective": {{
    "arguments_made": ["arg1", "arg2", ...],
    "arguments_encountered": ["arg1", "arg2", ...],
    "interaction_quality": <float 0-1>,
    "validation_received": <float 0-1>,
    "conflict_level": <float 0-1>
  }}
}}"""

    EVALUATE_ARGUMENT_QUALITY = """Evaluate the quality of these arguments from {speaker_name}'s perspective.

LISTENER PROFILE:
- Education: {education}
- Openness to new ideas: {openness}/100
- Need for evidence: {conscientiousness}/100
- Current knowledge on topic: {knowledge}/100

ARGUMENTS PRESENTED:
{arguments}

Evaluate argument quality based on these weighted factors:

1. LOGICAL COHERENCE (25%)
   - Clear reasoning and structure
   - No contradictions or fallacies
   - Follows from premises to conclusions
   - Score higher if listener has high conscientiousness

2. EVIDENCE QUALITY (25%)
   - Specific facts, data, or examples
   - Credible sources mentioned
   - Personal experience (if relevant)
   - Score higher if listener has high conscientiousness or education

3. EMOTIONAL RESONANCE (25%)
   - Appeals to listener's values
   - Acknowledges emotional stakes
   - Uses relatable examples
   - Score higher if listener has high emotionality

4. NOVELTY & INSIGHT (25%)
   - New perspectives offered
   - Challenges assumptions constructively
   - Creative solutions proposed
   - Score higher if listener has high openness

Adjust the final score based on:
- If listener has LOW openness (<40): Reduce score by 20% for arguments that challenge their worldview
- If listener has HIGH knowledge (>70): Reduce score by 15% for oversimplified arguments
- If listener has LOW knowledge (<30): Reduce score by 10% for overly technical arguments

Respond with ONLY a number between 0 and 1 representing the overall quality score."""

    @staticmethod
    def format_personality_description(personality_traits, behavior_implications):
        """Format personality traits into natural language description."""
        descriptions = []
        
        trait_descriptions = {
            'honesty_humility': {
                'very_high': "exceptionally honest and humble",
                'high': "sincere and modest", 
                'medium': "reasonably honest",
                'low': "somewhat self-interested",
                'very_low': "highly self-focused and strategic"
            },
            'emotionality': {
                'very_high': "highly sensitive and emotionally reactive",
                'high': "emotionally expressive and empathetic",
                'medium': "moderately emotional",
                'low': "emotionally stable and independent",
                'very_low': "exceptionally calm and detached"
            },
            'extraversion': {
                'very_high': "extremely outgoing and energetic",
                'high': "social and assertive",
                'medium': "moderately social",
                'low': "reserved and quiet",
                'very_low': "highly introverted and solitary"
            },
            'agreeableness': {
                'very_high': "exceptionally cooperative and gentle",
                'high': "patient and accommodating",
                'medium': "reasonably agreeable",
                'low': "direct and competitive",
                'very_low': "highly critical and confrontational"
            },
            'conscientiousness': {
                'very_high': "extremely organized and disciplined",
                'high': "diligent and careful",
                'medium': "moderately organized",
                'low': "flexible and spontaneous",
                'very_low': "highly impulsive and carefree"
            },
            'openness': {
                'very_high': "exceptionally creative and unconventional",
                'high': "curious and imaginative",
                'medium': "moderately open to new experiences",
                'low': "practical and traditional",
                'very_low': "highly conventional and routine-oriented"
            }
        }
        
        for trait_name, impl in behavior_implications.items():
            level = impl['level']
            desc = trait_descriptions.get(trait_name, {}).get(level, "moderate")
            descriptions.append(f"You are {desc}")
        
        return ". ".join(descriptions) + "."
    
