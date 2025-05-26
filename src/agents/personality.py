"""Personality-behavior ontology for mapping HEXACO traits to agent behaviors."""
from typing import Dict, Callable
from .profile import PersonalityTraits


class PersonalityBehaviorOntology:
    """Defines how HEXACO traits influence agent behavior."""
    
    # Conversation style mappings with more granular bands
    CONVERSATION_PATTERNS = {
        'honesty_humility': {
            'very_high': {  # 81-100
                'communication_style': 'radically transparent, self-deprecating',
                'topic_approach': 'admits uncertainty readily, credits others',
                'conflict_style': 'takes blame even when not at fault',
                'persuasion_ethics': 'refuses any form of influence'
            },
            'high': {  # 61-80
                'communication_style': 'straightforward, sincere',
                'topic_approach': 'factual, avoids exaggeration',
                'conflict_style': 'seeks fair resolution',
                'persuasion_ethics': 'avoids manipulation'
            },
            'medium': {  # 41-60
                'communication_style': 'balanced directness',
                'topic_approach': 'mostly accurate with occasional emphasis',
                'conflict_style': 'pragmatic compromise',
                'persuasion_ethics': 'mild influence acceptable'
            },
            'low': {  # 21-40
                'communication_style': 'strategic, may embellish',
                'topic_approach': 'focuses on personal gain',
                'conflict_style': 'winning-oriented',
                'persuasion_ethics': 'ends justify means'
            },
            'very_low': {  # 0-20
                'communication_style': 'manipulative, deceptive when beneficial',
                'topic_approach': 'twists facts to advantage',
                'conflict_style': 'dominance at any cost',
                'persuasion_ethics': 'no ethical constraints'
            }
        },
        'emotionality': {
            'very_high': {  # 81-100
                'stress_response': 'overwhelmed quickly, needs constant support',
                'empathy_expression': 'absorbs others emotions completely',
                'vulnerability': 'overshares personal anxieties',
                'decision_style': 'paralyzed by emotional implications'
            },
            'high': {  # 61-80
                'stress_response': 'easily overwhelmed, seeks support',
                'empathy_expression': 'highly responsive to others emotions',
                'vulnerability': 'openly shares fears and concerns',
                'decision_style': 'weighs emotional consequences heavily'
            },
            'medium': {  # 41-60
                'stress_response': 'moderate stress tolerance',
                'empathy_expression': 'balanced emotional awareness',
                'vulnerability': 'selective emotional sharing',
                'decision_style': 'considers both logic and feelings'
            },
            'low': {  # 21-40
                'stress_response': 'remains calm under pressure',
                'empathy_expression': 'less affected by others emotions',
                'vulnerability': 'rarely shares personal concerns',
                'decision_style': 'logic-focused, pragmatic'
            },
            'very_low': {  # 0-20
                'stress_response': 'unaffected by extreme stress',
                'empathy_expression': 'emotionally detached from others',
                'vulnerability': 'never reveals weaknesses',
                'decision_style': 'purely rational, dismisses emotions'
            }
        },
        'extraversion': {
            'very_high': {  # 81-100
                'interaction_energy': 'compulsively seeks conversations',
                'speaking_ratio': 'monopolizes conversation',
                'topic_breadth': 'jumps between topics rapidly',
                'social_fatigue': 'becomes restless without interaction'
            },
            'high': {  # 61-80
                'interaction_energy': 'seeks out conversations',
                'speaking_ratio': 'dominates conversation time',
                'topic_breadth': 'introduces many topics',
                'social_fatigue': 'energized by interaction'
            },
            'medium': {  # 41-60
                'interaction_energy': 'comfortable initiating or responding',
                'speaking_ratio': 'balanced speaking and listening',
                'topic_breadth': 'moderate topic range',
                'social_fatigue': 'neutral energy from interaction'
            },
            'low': {  # 21-40
                'interaction_energy': 'selective engagement',
                'speaking_ratio': 'listens more than speaks',
                'topic_breadth': 'focuses on few deep topics',
                'social_fatigue': 'drained by extended interaction'
            },
            'very_low': {  # 0-20
                'interaction_energy': 'avoids conversations',
                'speaking_ratio': 'speaks only when necessary',
                'topic_breadth': 'single topic focus',
                'social_fatigue': 'exhausted by any interaction'
            }
        },
        'agreeableness': {
            'very_high': {  # 81-100
                'conflict_avoidance': 'avoids all conflict, over-accommodates',
                'criticism_style': 'cannot give negative feedback',
                'cooperation': 'sacrifices own needs completely',
                'trust_tendency': 'naive, easily exploited'
            },
            'high': {  # 61-80
                'conflict_avoidance': 'seeks harmony, compromises',
                'criticism_style': 'gentle, constructive',
                'cooperation': 'looks for win-win solutions',
                'trust_tendency': 'assumes good intentions'
            },
            'medium': {  # 41-60
                'conflict_avoidance': 'addresses issues diplomatically',
                'criticism_style': 'balanced and fair',
                'cooperation': 'reciprocal collaboration',
                'trust_tendency': 'cautiously optimistic'
            },
            'low': {  # 21-40
                'conflict_avoidance': 'comfortable with disagreement',
                'criticism_style': 'direct, potentially harsh',
                'cooperation': 'prioritizes personal goals',
                'trust_tendency': 'skeptical of others motives'
            },
            'very_low': {  # 0-20
                'conflict_avoidance': 'seeks confrontation',
                'criticism_style': 'brutally critical',
                'cooperation': 'exploits others weakness',
                'trust_tendency': 'paranoid, sees threats everywhere'
            }
        },
        'conscientiousness': {
            'very_high': {  # 81-100
                'argument_structure': 'obsessively structured, rigid',
                'fact_checking': 'paralyzing perfectionism',
                'commitment': 'inflexible even when harmful',
                'time_awareness': 'anxiously clock-watches'
            },
            'high': {  # 61-80
                'argument_structure': 'organized, systematic',
                'fact_checking': 'careful about accuracy',
                'commitment': 'follows through on statements',
                'time_awareness': 'respects conversation bounds'
            },
            'medium': {  # 41-60
                'argument_structure': 'reasonably organized',
                'fact_checking': 'generally accurate',
                'commitment': 'keeps important promises',
                'time_awareness': 'aware but flexible'
            },
            'low': {  # 21-40
                'argument_structure': 'spontaneous, may ramble',
                'fact_checking': 'makes broad generalizations',
                'commitment': 'flexible with promises',
                'time_awareness': 'loses track of time'
            },
            'very_low': {  # 0-20
                'argument_structure': 'chaotic, contradictory',
                'fact_checking': 'indifferent to accuracy',
                'commitment': 'promises mean nothing',
                'time_awareness': 'oblivious to time'
            }
        },
        'openness': {
            'very_high': {  # 81-100
                'idea_receptivity': 'obsessed with novelty',
                'opinion_flexibility': 'changes views constantly',
                'curiosity': 'questions everything relentlessly',
                'abstraction': 'lost in theoretical abstractions'
            },
            'high': {  # 61-80
                'idea_receptivity': 'eager to explore new concepts',
                'opinion_flexibility': 'willing to change views',
                'curiosity': 'asks probing questions',
                'abstraction': 'enjoys theoretical discussions'
            },
            'medium': {  # 41-60
                'idea_receptivity': 'open but discerning',
                'opinion_flexibility': 'changes views with evidence',
                'curiosity': 'selectively curious',
                'abstraction': 'balances theory and practice'
            },
            'low': {  # 21-40
                'idea_receptivity': 'prefers familiar concepts',
                'opinion_flexibility': 'maintains established views',
                'curiosity': 'focuses on practical matters',
                'abstraction': 'prefers concrete examples'
            },
            'very_low': {  # 0-20
                'idea_receptivity': 'rejects anything new',
                'opinion_flexibility': 'views set in stone',
                'curiosity': 'actively incurious',
                'abstraction': 'only literal thinking'
            }
        }
    }
    
    # Opinion change receptivity modifiers
    INFLUENCE_MODIFIERS = {
        'honesty_humility': {
            'source_credibility_weight': lambda h: 0.5 + (h/100) * 0.5,  # High H values credibility more
            'emotional_appeal_resistance': lambda h: h/100  # High H resists emotional manipulation
        },
        'emotionality': {
            'emotional_contagion': lambda e: e/100,  # High E catches others' emotions
            'stress_opinion_volatility': lambda e: 0.5 + (e/100) * 0.5  # High E changes opinions under stress
        },
        'extraversion': {
            'social_proof_sensitivity': lambda e: 0.3 + (e/100) * 0.4,  # High E influenced by group
            'conversation_engagement': lambda e: 0.5 + (e/100) * 0.5  # High E more engaged
        },
        'agreeableness': {
            'conflict_opinion_shift': lambda a: a/100,  # High A shifts to reduce conflict
            'empathy_driven_change': lambda a: 0.3 + (a/100) * 0.5  # High A influenced by others' needs
        },
        'conscientiousness': {
            'evidence_requirement': lambda c: 0.3 + (c/100) * 0.7,  # High C needs strong evidence
            'consistency_pressure': lambda c: c/100  # High C resists contradicting past positions
        },
        'openness': {
            'novelty_bonus': lambda o: (o/100) * 0.5,  # High O gives weight to new ideas
            'certainty_flexibility': lambda o: 1 - (o/100) * 0.5  # High O has lower certainty barriers
        }
    }
    
    @staticmethod
    def get_conversation_intent_probability(personality: PersonalityTraits) -> Dict[str, float]:
        """Calculate probability of each conversation intent based on personality."""
        return {
            'learn': 0.2 + (personality.openness/100) * 0.4 + (100-personality.emotionality)/100 * 0.2,
            'persuade': 0.1 + (personality.extraversion/100) * 0.3 + (100-personality.agreeableness)/100 * 0.2,
            'validate': 0.1 + (personality.emotionality/100) * 0.4 + (100-personality.openness)/100 * 0.2,
            'bond': 0.2 + (personality.agreeableness/100) * 0.4 + (personality.extraversion/100) * 0.2,
            'debate': 0.1 + (personality.openness/100) * 0.3 + (100-personality.agreeableness)/100 * 0.2,
            'explore': 0.2 + (personality.openness/100) * 0.3 + (personality.conscientiousness/100) * 0.1
        }
    
    @staticmethod
    def get_trait_level(trait_value: int) -> str:
        """Convert numeric trait value to level descriptor."""
        if trait_value <= 20:
            return 'very_low'
        elif trait_value <= 40:
            return 'low'
        elif trait_value <= 60:
            return 'medium'
        elif trait_value <= 80:
            return 'high'
        else:
            return 'very_high'
    
    @classmethod
    def get_behavior_pattern(cls, trait_name: str, trait_value: int) -> Dict[str, str]:
        """Get behavior patterns for a trait at a given level."""
        level = cls.get_trait_level(trait_value)
        return cls.CONVERSATION_PATTERNS.get(trait_name, {}).get(level, {})
    
    @classmethod
    def get_influence_modifier(cls, trait_name: str, modifier_type: str, trait_value: int) -> float:
        """Get influence modifier value for a trait."""
        trait_modifiers = cls.INFLUENCE_MODIFIERS.get(trait_name, {})
        modifier_func = trait_modifiers.get(modifier_type)
        if modifier_func and callable(modifier_func):
            return modifier_func(trait_value)
        return 1.0  # Default neutral modifier
    
    @classmethod
    def describe_personality_implications(cls, personality: PersonalityTraits) -> Dict[str, Dict[str, str]]:
        """Generate comprehensive behavioral implications from personality profile."""
        implications = {}
        
        trait_names = ['honesty_humility', 'emotionality', 'extraversion', 
                      'agreeableness', 'conscientiousness', 'openness']
        
        for trait_name in trait_names:
            trait_value = getattr(personality, trait_name)
            implications[trait_name] = {
                'level': cls.get_trait_level(trait_value),
                'behaviors': cls.get_behavior_pattern(trait_name, trait_value)
            }
        
        return implications