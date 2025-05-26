"""Agent data structures for the social dynamics modeling framework."""
from dataclasses import dataclass, field
from typing import Dict, List
from datetime import datetime
import uuid


@dataclass
class PersonalityTraits:
    """HEXACO personality model with behavioral implications."""
    honesty_humility: int      # 0-100: Sincerity, fairness, modesty
    emotionality: int          # 0-100: Emotional reactivity, anxiety
    extraversion: int          # 0-100: Sociability, energy, boldness  
    agreeableness: int         # 0-100: Patience, tolerance, gentleness
    conscientiousness: int     # 0-100: Organization, diligence, perfectionism
    openness: int             # 0-100: Curiosity, creativity, unconventionality

    def __post_init__(self):
        """Validate trait values are within bounds."""
        for trait_name in ['honesty_humility', 'emotionality', 'extraversion', 
                          'agreeableness', 'conscientiousness', 'openness']:
            value = getattr(self, trait_name)
            if not 0 <= value <= 100:
                raise ValueError(f"{trait_name} must be between 0 and 100, got {value}")


@dataclass
class Background:
    """Stable demographic and life history factors."""
    name: str = ""  # Full name of the agent
    age: int = 30
    occupation: str = "Professional"
    education_level: str = "bachelors"  # high_school, bachelors, masters, phd
    education_field: str = "General Studies"  # e.g., "computer science", "literature"
    socioeconomic_tags: List[str] = field(default_factory=list)  # e.g., ["middle-class", "suburban"]
    relationship_tags: List[str] = field(default_factory=list)    # e.g., ["married", "parent-of-two"]
    cultural_tags: List[str] = field(default_factory=list)        # e.g., ["progressive", "religious"]

    def __post_init__(self):
        """Validate background values."""
        if not 18 <= self.age <= 100:
            raise ValueError(f"Age must be between 18 and 100, got {self.age}")
        
        valid_education_levels = ['no_high_school', 'high_school', 'some_college', 
                                 'associates', 'bachelors', 'masters', 'phd']
        if self.education_level not in valid_education_levels:
            raise ValueError(f"Invalid education level: {self.education_level}. Valid options: {valid_education_levels}")


@dataclass
class EmotionalBaseline:
    """Stable emotional tendencies."""
    dispositional_affect: int    # -50 to +50: General mood tendency
    stress_tolerance: int        # 0-100: Resilience to stressors
    social_confidence: int       # 0-100: Comfort in social situations
    self_efficacy: int          # 0-100: Belief in own capabilities

    def __post_init__(self):
        """Validate emotional baseline values."""
        if not -50 <= self.dispositional_affect <= 50:
            raise ValueError(f"Dispositional affect must be between -50 and 50")
        
        for attr_name in ['stress_tolerance', 'social_confidence', 'self_efficacy']:
            value = getattr(self, attr_name)
            if not 0 <= value <= 100:
                raise ValueError(f"{attr_name} must be between 0 and 100, got {value}")


@dataclass
class EmotionalState:
    """Current dynamic emotional state."""
    arousal: int = 50           # 0-100: Calm to excited
    valence: int = 0            # -50 to +50: Negative to positive
    anxiety: int = 25           # 0-100: Current worry level
    confidence: int = 50        # 0-100: Current self-assurance
    social_energy: int = 50     # 0-100: Desire for interaction
    cognitive_load: int = 25    # 0-100: Mental fatigue

    def __post_init__(self):
        """Validate emotional state values."""
        if not -50 <= self.valence <= 50:
            raise ValueError(f"Valence must be between -50 and 50")
        
        for attr_name in ['arousal', 'anxiety', 'confidence', 'social_energy', 'cognitive_load']:
            value = getattr(self, attr_name)
            if not 0 <= value <= 100:
                raise ValueError(f"{attr_name} must be between 0 and 100, got {value}")


@dataclass
class Opinion:
    """Opinion on a specific topic."""
    position: int              # -100 to +100: Topic-specific scale
    certainty: int             # 0-100: Confidence in position
    importance: int            # 0-100: Personal relevance
    knowledge: int             # 0-100: Perceived expertise
    emotional_charge: int      # 0-100: Emotional investment

    def __post_init__(self):
        """Validate opinion values."""
        if not -100 <= self.position <= 100:
            raise ValueError(f"Position must be between -100 and 100")
        
        for attr_name in ['certainty', 'importance', 'knowledge', 'emotional_charge']:
            value = getattr(self, attr_name)
            if not 0 <= value <= 100:
                raise ValueError(f"{attr_name} must be between 0 and 100, got {value}")


@dataclass
class Agent:
    """Complete agent profile."""
    # Unique identifier
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Stable traits (immutable after creation)
    name: str = ""
    personality: PersonalityTraits = field(default_factory=lambda: PersonalityTraits(50, 50, 50, 50, 50, 50))
    background: Background = field(default_factory=lambda: Background(
        30, "Unknown", "bachelors", "general", [], [], []))
    emotional_baseline: EmotionalBaseline = field(default_factory=lambda: EmotionalBaseline(0, 50, 50, 50))
    biography: str = ""
    conversation_style: str = ""  # Detailed description of how this agent speaks
    
    # Dynamic state
    emotional_state: EmotionalState = field(default_factory=EmotionalState)
    opinions: Dict[str, Opinion] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_interaction: datetime = None

    def update_emotional_state(self, **kwargs):
        """Update emotional state with bounds checking."""
        for key, value in kwargs.items():
            if hasattr(self.emotional_state, key):
                if key == 'valence':
                    new_value = max(-50, min(50, value))
                else:
                    new_value = max(0, min(100, value))
                setattr(self.emotional_state, key, new_value)

    def update_opinion(self, topic: str, **kwargs):
        """Update opinion on a topic with bounds checking."""
        if topic not in self.opinions:
            raise ValueError(f"Agent has no opinion on topic: {topic}")
        
        opinion = self.opinions[topic]
        for key, value in kwargs.items():
            if hasattr(opinion, key):
                if key == 'position':
                    new_value = max(-100, min(100, value))
                else:
                    new_value = max(0, min(100, value))
                setattr(opinion, key, new_value)