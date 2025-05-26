"""LLM-based agent generation system."""
from typing import Dict, Optional, List
import json
import logging
from ..llm.client import LLMClient
from ..llm.prompts import PromptTemplates
from .profile import (
    Agent, PersonalityTraits, Background, EmotionalBaseline, 
    EmotionalState, Opinion
)
from .personality import PersonalityBehaviorOntology
from .name_generator import NameGenerator

logger = logging.getLogger('dynavox.agents')


class AgentGenerator:
    """Generates complete agent profiles using LLM."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.ontology = PersonalityBehaviorOntology()
        self.prompts = PromptTemplates()
    
    def generate_agent(self, constraints: Optional[Dict] = None) -> Agent:
        """Generate a complete agent profile."""
        constraints = constraints or {}
        logger.debug(f"Generating new agent with constraints: {constraints}")
        
        # Step 1: Generate personality traits
        logger.debug("Step 1: Generating personality traits")
        personality = self._generate_personality(constraints)
        logger.debug(f"Generated personality: H={personality.honesty_humility}, E={personality.emotionality}, "
                    f"X={personality.extraversion}, A={personality.agreeableness}, "
                    f"C={personality.conscientiousness}, O={personality.openness}")
        
        # Step 2: Generate consistent background
        logger.debug("Step 2: Generating background")
        background = self._generate_background(personality, constraints)
        logger.debug(f"Generated background: {background.name}, {background.age}yo {background.occupation}")
        
        # Step 3: Derive emotional baseline
        logger.debug("Step 3: Generating emotional baseline")
        emotional_baseline = self._generate_emotional_baseline(personality, background)
        logger.debug(f"Generated emotional baseline: affect={emotional_baseline.dispositional_affect}, "
                    f"stress_tolerance={emotional_baseline.stress_tolerance}")
        
        # Step 4: Generate biography
        logger.debug("Step 4: Generating biography")
        biography = self._generate_biography(personality, background, emotional_baseline)
        logger.debug(f"Generated biography of {len(biography.split())} words")
        
        # Step 5: Generate conversation style
        logger.debug("Step 5: Generating conversation style")
        conversation_style = self._generate_conversation_style(personality, background, emotional_baseline)
        logger.debug(f"Generated conversation style of {len(conversation_style.split())} words")
        
        # Step 6: Initialize opinions for pre-selected topics
        topics = constraints.get('topics', [])
        logger.debug(f"Step 6: Initializing opinions for {len(topics)} topics")
        opinions = self._initialize_opinions(personality, background, biography, constraints)
        for topic, opinion in opinions.items():
            logger.debug(f"  Opinion on {topic}: position={opinion.position}, certainty={opinion.certainty}")
        
        # Step 7: Set initial emotional state
        logger.debug("Step 7: Setting initial emotional state")
        emotional_state = self._set_initial_emotional_state(emotional_baseline, background)
        logger.debug(f"Initial emotional state: valence={emotional_state.valence}, "
                    f"anxiety={emotional_state.anxiety}, social_energy={emotional_state.social_energy}")
        
        # Use the name from background
        name = background.name if background.name else f"Agent_{id(background) % 10000}"
        logger.info(f"Successfully generated agent: {name}")
        
        return Agent(
            name=name,
            personality=personality,
            background=background,
            emotional_baseline=emotional_baseline,
            biography=biography,
            conversation_style=conversation_style,
            emotional_state=emotional_state,
            opinions=opinions
        )
    
    def _generate_personality(self, constraints: Dict) -> PersonalityTraits:
        """Generate HEXACO traits using LLM."""
        constraint_text = ""
        if 'personality_bias' in constraints:
            constraint_text = f"Bias the personality toward: {constraints['personality_bias']}"
            
        prompt = self.prompts.GENERATE_PERSONALITY.format(constraints=constraint_text)
        
        try:
            traits_dict = self.llm.generate_json(prompt)
            return PersonalityTraits(**traits_dict)
        except Exception as e:
            # Fallback to default balanced personality
            print(f"Warning: Failed to generate personality: {e}. Using defaults.")
            return PersonalityTraits(
                honesty_humility=50 + (hash(str(constraints)) % 20 - 10),
                emotionality=50 + (hash(str(constraints) + "e") % 20 - 10),
                extraversion=50 + (hash(str(constraints) + "ex") % 20 - 10),
                agreeableness=50 + (hash(str(constraints) + "a") % 20 - 10),
                conscientiousness=50 + (hash(str(constraints) + "c") % 20 - 10),
                openness=50 + (hash(str(constraints) + "o") % 20 - 10)
            )
    
    def _generate_background(self, personality: PersonalityTraits, constraints: Dict) -> Background:
        """Generate background consistent with personality."""
        constraint_text = ""
        if 'demographics' in constraints and constraints['demographics']:
            demo = constraints['demographics']
            constraint_parts = []
            
            if 'age_range' in demo:
                constraint_parts.append(f"Age should be between {demo['age_range'][0]} and {demo['age_range'][1]}")
            
            if 'education_preference' in demo:
                constraint_parts.append(f"Education level should be: {demo['education_preference']}")
            
            if 'socioeconomic_preference' in demo:
                constraint_parts.append(f"Socioeconomic status should reflect: {demo['socioeconomic_preference']}")
            
            if 'occupation_category' in demo:
                from ..config import AGENT_GENERATION_PARAMS
                occupations = AGENT_GENERATION_PARAMS['occupation_categories'].get(demo['occupation_category'], [])
                if occupations:
                    constraint_parts.append(f"Occupation should be one of: {', '.join(occupations[:3])}")
            
            if 'location_preference' in demo:
                constraint_parts.append(f"Background should reflect {demo['location_preference']} life")
            
            if 'relationship_preference' in demo:
                rel_map = {
                    'single': 'single, never married, or dating',
                    'partnered': 'married or in long-term relationship',
                    'family_focused': 'parent with children',
                    'complex': 'divorced, widowed, or separated'
                }
                constraint_parts.append(f"Relationship status: {rel_map.get(demo['relationship_preference'], 'any')}")
            
            constraint_text = "Additional constraints: " + "; ".join(constraint_parts)
            
        prompt = self.prompts.GENERATE_BACKGROUND.format(
            honesty_humility=personality.honesty_humility,
            emotionality=personality.emotionality,
            extraversion=personality.extraversion,
            agreeableness=personality.agreeableness,
            conscientiousness=personality.conscientiousness,
            openness=personality.openness,
            constraints=constraint_text
        )
        
        try:
            background_dict = self.llm.generate_json(prompt)
            
            # Create background without name first
            background = Background(
                age=background_dict['age'],
                occupation=background_dict['occupation'],
                education_level=background_dict['education_level'],
                education_field=background_dict['education_field'],
                socioeconomic_tags=background_dict.get('socioeconomic_tags', []),
                relationship_tags=background_dict.get('relationship_tags', []),
                cultural_tags=background_dict.get('cultural_tags', [])
            )
            
            # Generate culturally appropriate name
            first_name, last_name = NameGenerator.generate_name(
                background.cultural_tags,
                background.age,
                background.socioeconomic_tags
            )
            background.name = NameGenerator.format_full_name(
                first_name, last_name, background.cultural_tags
            )
            
            return background
            
        except Exception as e:
            # Fallback background
            print(f"Warning: Failed to generate background: {e}. Using defaults.")
            background = Background(
                age=35,
                occupation="Professional",
                education_level="bachelors",
                education_field="General Studies",
                socioeconomic_tags=["middle-class"],
                relationship_tags=["single"],
                cultural_tags=["moderate"]
            )
            
            # Generate name for fallback
            first_name, last_name = NameGenerator.generate_name(
                background.cultural_tags,
                background.age,
                background.socioeconomic_tags
            )
            background.name = NameGenerator.format_full_name(
                first_name, last_name, background.cultural_tags
            )
            
            return background
    
    def _generate_emotional_baseline(self, personality: PersonalityTraits, 
                                   background: Background) -> EmotionalBaseline:
        """Generate emotional baseline from personality and background."""
        relationship_status = ', '.join(background.relationship_tags) if background.relationship_tags else "unspecified"
        socioeconomic = ', '.join(background.socioeconomic_tags) if background.socioeconomic_tags else "unspecified"
        
        prompt = self.prompts.GENERATE_EMOTIONAL_BASELINE.format(
            honesty_humility=personality.honesty_humility,
            emotionality=personality.emotionality,
            extraversion=personality.extraversion,
            agreeableness=personality.agreeableness,
            conscientiousness=personality.conscientiousness,
            openness=personality.openness,
            age=background.age,
            occupation=background.occupation,
            relationship_status=relationship_status,
            socioeconomic=socioeconomic
        )
        
        try:
            baseline_dict = self.llm.generate_json(prompt)
            return EmotionalBaseline(**baseline_dict)
        except Exception as e:
            # Calculate reasonable defaults based on personality
            print(f"Warning: Failed to generate emotional baseline: {e}. Using calculated defaults.")
            
            # Higher emotionality = lower stress tolerance
            stress_tolerance = max(20, min(80, 70 - personality.emotionality // 2))
            
            # Higher extraversion = higher social confidence
            social_confidence = max(20, min(80, 30 + personality.extraversion // 2))
            
            # Higher conscientiousness and lower emotionality = higher self-efficacy
            self_efficacy = max(20, min(80, 40 + personality.conscientiousness // 3 - personality.emotionality // 4))
            
            # Dispositional affect influenced by multiple factors
            dispositional_affect = 0
            if personality.extraversion > 60:
                dispositional_affect += 10
            if personality.emotionality > 70:
                dispositional_affect -= 15
            if personality.openness > 60:
                dispositional_affect += 5
            
            return EmotionalBaseline(
                dispositional_affect=dispositional_affect,
                stress_tolerance=stress_tolerance,
                social_confidence=social_confidence,
                self_efficacy=self_efficacy
            )
    
    def _generate_biography(self, personality: PersonalityTraits, 
                          background: Background, 
                          emotional_baseline: EmotionalBaseline) -> str:
        """Generate rich biographical narrative."""
        mood_tendency = "positive" if emotional_baseline.dispositional_affect > 0 else "negative"
        
        prompt = self.prompts.GENERATE_BIOGRAPHY.format(
            name=background.name if background.name else "This person",
            age=background.age,
            occupation=background.occupation,
            education_level=background.education_level,
            education_field=background.education_field,
            honesty_humility=personality.honesty_humility,
            emotionality=personality.emotionality,
            extraversion=personality.extraversion,
            agreeableness=personality.agreeableness,
            conscientiousness=personality.conscientiousness,
            openness=personality.openness,
            socioeconomic=', '.join(background.socioeconomic_tags),
            relationships=', '.join(background.relationship_tags),
            cultural=', '.join(background.cultural_tags),
            mood_tendency=f"{mood_tendency} ({abs(emotional_baseline.dispositional_affect)}/50)",
            stress_tolerance=emotional_baseline.stress_tolerance,
            social_confidence=emotional_baseline.social_confidence,
            self_efficacy=emotional_baseline.self_efficacy
        )
        
        try:
            biography = self.llm.generate(prompt)
            return biography
        except Exception as e:
            # Generate a basic biography
            print(f"Warning: Failed to generate biography: {e}. Using template.")
            return self._generate_template_biography(personality, background, emotional_baseline)
    
    def _generate_conversation_style(self, personality: PersonalityTraits,
                                   background: Background,
                                   emotional_baseline: EmotionalBaseline) -> str:
        """Generate detailed conversation style description."""
        prompt = self.prompts.GENERATE_CONVERSATION_STYLE.format(
            name=background.name if background.name else "This person",
            age=background.age,
            occupation=background.occupation,
            education_level=background.education_level,
            education_field=background.education_field,
            honesty_humility=personality.honesty_humility,
            emotionality=personality.emotionality,
            extraversion=personality.extraversion,
            agreeableness=personality.agreeableness,
            conscientiousness=personality.conscientiousness,
            openness=personality.openness,
            socioeconomic=', '.join(background.socioeconomic_tags),
            cultural=', '.join(background.cultural_tags),
            social_confidence=emotional_baseline.social_confidence,
            self_efficacy=emotional_baseline.self_efficacy
        )
        
        try:
            conversation_style = self.llm.generate(prompt)
            return conversation_style
        except Exception as e:
            # Generate a basic conversation style
            print(f"Warning: Failed to generate conversation style: {e}. Using template.")
            return self._generate_template_conversation_style(personality, background, emotional_baseline)
    
    def _initialize_opinions(self, personality: PersonalityTraits,
                           background: Background,
                           biography: str,
                           constraints: Dict) -> Dict[str, Opinion]:
        """Initialize opinions on pre-selected topics."""
        topics = constraints.get('topics', ['climate_change', 'remote_work', 'universal_healthcare'])
        opinions = {}
        
        # Use name from background
        name = background.name if background.name else "This person"
        
        for topic in topics:
            prompt = self.prompts.INITIALIZE_OPINION.format(
                topic=topic,
                name=name,
                occupation=background.occupation,
                education_level=background.education_level,
                education_field=background.education_field,
                cultural=', '.join(background.cultural_tags),
                openness=personality.openness,
                conscientiousness=personality.conscientiousness,
                agreeableness=personality.agreeableness,
                biography_excerpt=biography[:500] if biography else "No biography available"
            )
            
            try:
                opinion_dict = self.llm.generate_json(prompt)
                opinions[topic] = Opinion(**opinion_dict)
            except Exception as e:
                # Generate reasonable defaults based on personality
                print(f"Warning: Failed to generate opinion for {topic}: {e}. Using defaults.")
                opinions[topic] = self._generate_default_opinion(topic, personality, background)
        
        return opinions
    
    def _set_initial_emotional_state(self, emotional_baseline: EmotionalBaseline, 
                                   background: Background) -> EmotionalState:
        """Set initial emotional state based on baseline and current life situation."""
        # Start from neutral/baseline-influenced state
        arousal = 50
        valence = emotional_baseline.dispositional_affect // 2  # Moderate the baseline
        anxiety = 30 - emotional_baseline.stress_tolerance // 4  # Lower stress tolerance = higher baseline anxiety
        confidence = 30 + emotional_baseline.self_efficacy // 3
        social_energy = 40 + emotional_baseline.social_confidence // 4
        cognitive_load = 25
        
        # Adjust based on life circumstances
        if 'unemployed' in background.socioeconomic_tags:
            anxiety += 15
            confidence -= 10
            valence -= 10
        
        if 'divorced' in background.relationship_tags or 'widowed' in background.relationship_tags:
            valence -= 15
            social_energy -= 10
        
        if 'parent' in ' '.join(background.relationship_tags):
            cognitive_load += 15
            arousal += 10
        
        # Ensure all values are within bounds
        return EmotionalState(
            arousal=max(0, min(100, arousal)),
            valence=max(-50, min(50, valence)),
            anxiety=max(0, min(100, anxiety)),
            confidence=max(0, min(100, confidence)),
            social_energy=max(0, min(100, social_energy)),
            cognitive_load=max(0, min(100, cognitive_load))
        )
    
    def _generate_default_opinion(self, topic: str, personality: PersonalityTraits, 
                                background: Background) -> Opinion:
        """Generate a default opinion based on personality and background."""
        # Base position influenced by personality
        position = 0
        
        # Topic-specific adjustments
        if topic == 'climate_change':
            if personality.openness > 60:
                position += 30
            if 'progressive' in background.cultural_tags:
                position += 20
            if personality.conscientiousness > 70:
                position += 15
                
        elif topic == 'universal_healthcare':
            if personality.agreeableness > 60:
                position += 25
            if 'progressive' in background.cultural_tags:
                position += 30
            if personality.emotionality > 60:
                position += 15
                
        elif topic == 'remote_work':
            if personality.openness > 60:
                position += 20
            if personality.extraversion < 40:
                position += 25  # Introverts prefer remote work
            if 'tech' in background.occupation.lower():
                position += 30
        
        # Certainty based on conscientiousness and education
        certainty = 50
        if personality.conscientiousness > 70:
            certainty += 15
        if background.education_level in ['masters', 'phd']:
            certainty += 10
            
        # Importance varies by topic relevance to person
        importance = 40
        if topic == 'climate_change' and background.age < 35:
            importance += 20
        if topic == 'universal_healthcare' and personality.emotionality > 60:
            importance += 15
        if topic == 'remote_work' and 'parent' in ' '.join(background.relationship_tags):
            importance += 20
            
        # Knowledge based on education and occupation relevance
        knowledge = 40
        if background.education_level in ['masters', 'phd']:
            knowledge += 20
        if personality.openness > 70:
            knowledge += 10
            
        # Emotional charge based on personality
        emotional_charge = 30
        if personality.emotionality > 60:
            emotional_charge += 20
        if importance > 60:
            emotional_charge += 15
        
        return Opinion(
            position=max(-100, min(100, position)),
            certainty=max(0, min(100, certainty)),
            importance=max(0, min(100, importance)),
            knowledge=max(0, min(100, knowledge)),
            emotional_charge=max(0, min(100, emotional_charge))
        )
    
    def _generate_template_biography(self, personality: PersonalityTraits,
                                   background: Background,
                                   emotional_baseline: EmotionalBaseline) -> str:
        """Generate a basic template biography when LLM fails."""
        name = background.name if background.name else 'This individual'
        
        # Personality description
        personality_desc = []
        if personality.extraversion > 60:
            personality_desc.append("outgoing and energetic")
        elif personality.extraversion < 40:
            personality_desc.append("reserved and introspective")
            
        if personality.agreeableness > 60:
            personality_desc.append("cooperative and compassionate")
        elif personality.agreeableness < 40:
            personality_desc.append("direct and competitive")
            
        if personality.openness > 60:
            personality_desc.append("creative and curious")
        elif personality.openness < 40:
            personality_desc.append("practical and traditional")
        
        personality_str = " and ".join(personality_desc) if personality_desc else "balanced"
        
        biography = f"""{name} is a {background.age}-year-old {background.occupation} with a {background.education_level} degree in {background.education_field}. 
        
Growing up in a {', '.join(background.socioeconomic_tags)} environment with {', '.join(background.cultural_tags)} values, {name} developed a {personality_str} personality that has guided their life choices.

A defining moment came {'early in their career' if background.age < 35 else 'in their formative years'} when they {'discovered their passion for' if personality.openness > 60 else 'committed to'} their field. This led them to pursue {background.occupation}, where they {'thrive on innovation and creative problem-solving' if personality.openness > 60 else 'excel through dedication and consistency'}.

{name}'s core values center on {'personal growth and authentic self-expression' if personality.honesty_humility > 60 else 'practical achievement and strategic success'}. They believe strongly in {'collaborative approaches and mutual support' if personality.agreeableness > 60 else 'individual responsibility and merit-based outcomes'}.

Currently {', '.join(background.relationship_tags)}, {name} {'actively cultivates diverse social connections' if personality.extraversion > 60 else 'maintains a few deep, meaningful relationships'}. Their social circle consists mainly of {'fellow professionals and creative thinkers' if personality.openness > 60 else 'long-term friends and trusted colleagues'}.

In their free time, {name} {'explores new hobbies and cultural experiences' if personality.openness > 60 else 'enjoys familiar activities that provide relaxation'}. They're particularly drawn to {'intellectual pursuits and artistic expression' if personality.openness > 70 else 'practical hobbies and community involvement' if personality.conscientiousness > 70 else 'social activities and group events' if personality.extraversion > 70 else 'quiet, contemplative pastimes'}.

Their worldview is shaped by {'curiosity about human potential' if personality.openness > 60 else 'respect for tradition and proven methods'}, combined with a {'deep empathy for others' if personality.emotionality > 60 else 'pragmatic focus on results'}. This makes them someone who {'questions assumptions and seeks new perspectives' if personality.openness > 60 else 'values stability and clear principles'}."""
        
        return biography
    
    def _generate_template_conversation_style(self, personality: PersonalityTraits,
                                             background: Background,
                                             emotional_baseline: EmotionalBaseline) -> str:
        """Generate a template conversation style when LLM fails."""
        name = background.name if background.name else 'This person'
        
        # Base patterns on education and occupation
        education_patterns = {
            'phd': "precise academic language with careful qualifications",
            'masters': "professional vocabulary with clear structure",
            'bachelors': "educated but accessible language",
            'high_school': "straightforward, practical expressions"
        }
        
        vocab_level = education_patterns.get(background.education_level, "conversational language")
        
        # Personality-based patterns
        speech_length = "brief and to the point" if personality.extraversion < 40 else "moderately detailed" if personality.extraversion < 70 else "elaborative and expansive"
        
        thought_structure = "highly organized and linear" if personality.conscientiousness > 70 else "somewhat structured" if personality.conscientiousness > 40 else "free-flowing and spontaneous"
        
        emotional_expression = "openly expressive with frequent emotion words" if personality.emotionality > 70 else "moderate emotional expression" if personality.emotionality > 40 else "reserved and factual"
        
        listening_style = "active listening with validation" if personality.agreeableness > 70 else "attentive but neutral" if personality.agreeableness > 40 else "focused on own points"
        
        style = f"""{name} speaks with {vocab_level}, typical of their {background.occupation} background. Their responses tend to be {speech_length}, reflecting their {'introverted' if personality.extraversion < 40 else 'extraverted'} nature.

Their thought process is {thought_structure}. They {'frequently use metaphors and abstract concepts' if personality.openness > 70 else 'stick to concrete examples and proven ideas' if personality.openness < 30 else 'balance abstract and concrete thinking'}.

In terms of emotional expression, they are {emotional_expression}. When stressed, they {'become more talkative and seek reassurance' if personality.emotionality > 60 else 'become quieter and more analytical'}.

Their listening style is {listening_style}. They {'often interrupt with enthusiasm' if personality.extraversion > 70 and personality.conscientiousness < 40 else 'wait patiently for their turn' if personality.conscientiousness > 70 else 'engage in natural back-and-forth'}.

Common phrases include {'analytical terms like "evidence suggests" or "logically speaking"' if personality.openness > 60 and background.education_level in ['masters', 'phd'] else 'practical phrases like "in my experience" or "what works is"' if personality.conscientiousness > 60 else 'relational phrases like "I feel that" or "it seems to me"'}.

They {'rarely use humor' if personality.emotionality > 70 or personality.conscientiousness > 80 else 'use dry, intellectual humor' if personality.openness > 70 else 'employ warm, inclusive humor' if personality.agreeableness > 70 else 'occasionally use pointed humor'}."""
        
        return style