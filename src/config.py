"""Configuration settings for the social dynamics framework."""

# LLM Settings
# Available OpenAI models (Updated December 2024)
OPENAI_MODELS = {
    # Latest models
    "gpt-4o": "GPT-4o - Multimodal, fast, 83% cheaper than GPT-4 (128K context)",
    "gpt-4o-2024-11-20": "GPT-4o latest - Most recent version with improvements",
    "gpt-4o-mini": "GPT-4o Mini - Most cost-efficient, $0.15/1M input (128K context)",
    "gpt-4-turbo": "GPT-4 Turbo - High capability, $10/1M input (128K context)",
    "gpt-4-turbo-2024-04-09": "GPT-4 Turbo stable - April 2024 checkpoint",
    "gpt-4": "GPT-4 - Original, $30/1M input (8K context)",
    "gpt-4-32k": "GPT-4 32K - Extended context, $60/1M input",
    "gpt-3.5-turbo": "GPT-3.5 Turbo - Fast and cheap, $0.50/1M input",
    "gpt-3.5-turbo-0125": "GPT-3.5 Turbo latest - 16K context, optimized",
}

# Available Anthropic models (Updated December 2024)  
ANTHROPIC_MODELS = {
    "claude-opus-4": "Claude Opus 4 - Most powerful, $15/1M input (200K context)",
    "claude-sonnet-4": "Claude Sonnet 4 - Balanced, $3/1M input (200K context)",
    "claude-3-5-sonnet": "Claude 3.5 Sonnet - Upgraded, same price as v3 (200K context)",
    "claude-3-5-haiku": "Claude 3.5 Haiku - Fast & cheap, $0.80/1M input (200K context)",
}

# Default model selection
DEFAULT_OPENAI_MODEL = "gpt-4o"  # Updated to more cost-effective default
DEFAULT_ANTHROPIC_MODEL = "claude-3-5-sonnet"  # Balanced performance
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 1000

# Conversation Settings
MAX_CONVERSATION_TURNS = 20
MIN_CONVERSATION_TURNS = 3
CONCLUSION_KEYWORDS = ["goodbye", "bye", "talk later", "nice talking", 
                      "gotta go", "need to go", "see you", "take care"]

# Influence Parameters
INFLUENCE_THRESHOLD = 0.1  # Minimum influence to cause opinion change
MAX_POSITION_CHANGE = 20   # Maximum opinion shift in one conversation
CERTAINTY_DECAY_RATE = 0.05  # How much certainty decreases when challenged

# Personality Behavior Mappings
TRAIT_THRESHOLDS = {
    'very_low': 20,
    'low': 40,
    'medium': 60,
    'high': 80,
    'very_high': 100
}

# Emotional State Bounds
EMOTION_BOUNDS = {
    'arousal': (0, 100),
    'valence': (-50, 50),
    'anxiety': (0, 100),
    'confidence': (0, 100),
    'social_energy': (0, 100),
    'cognitive_load': (0, 100)
}

# Opinion Bounds
OPINION_BOUNDS = {
    'position': (-100, 100),
    'certainty': (0, 100),
    'importance': (0, 100),
    'knowledge': (0, 100),
    'emotional_charge': (0, 100)
}

# Topic Knowledge and Bias by Background
TOPIC_BACKGROUND_RELEVANCE = {
    'climate_change': {
        'high_knowledge': ['environmental_scientist', 'policy_analyst', 'teacher'],
        'personal_impact': ['farmer', 'construction_worker', 'coastal_resident'],
        'skepticism_tendency': ['coal_miner', 'oil_worker', 'rural', 'conservative']
    },
    'wealth_inequality': {
        'high_knowledge': ['economist', 'social_worker', 'politician'],
        'personal_impact': ['poverty', 'working-poor', 'unemployed', 'wealthy'],
        'strong_opinions': ['union_member', 'business_owner', 'minimum_wage_worker']
    },
    'ai_regulation': {
        'high_knowledge': ['tech_worker', 'researcher', 'policy_maker'],
        'job_threatened': ['driver', 'data_entry', 'factory_worker', 'cashier'],
        'minimal_exposure': ['rural', 'elderly', 'manual_labor']
    },
    'universal_healthcare': {
        'high_knowledge': ['healthcare_worker', 'insurance_agent', 'social_worker'],
        'personal_impact': ['chronic_illness', 'uninsured', 'self-employed'],
        'cost_concerns': ['small_business_owner', 'high_earner']
    },
    'education_reform': {
        'high_knowledge': ['teacher', 'parent', 'education_admin'],
        'personal_impact': ['student_debt', 'parent', 'recent_graduate'],
        'traditional_preference': ['rural', 'religious', 'older_generation']
    }
}

# Topic Definitions
STANDARD_TOPICS = {
    'climate_change': {
        'scale': (-100, 100),
        'labels': ('Climate skeptic', 'Climate activist'),
        'description': 'Views on climate change urgency and action'
    },
    'wealth_inequality': {
        'scale': (-100, 100),
        'labels': ('Pure meritocracy', 'Wealth redistribution'),
        'description': 'Views on economic inequality and solutions'
    },
    'ai_regulation': {
        'scale': (-100, 100),
        'labels': ('No regulation', 'Strict control'),
        'description': 'Views on AI governance and regulation'
    },
    'remote_work': {
        'scale': (-100, 100),
        'labels': ('Office only', 'Fully remote'),
        'description': 'Views on remote vs in-person work'
    },
    'universal_healthcare': {
        'scale': (-100, 100),
        'labels': ('Private only', 'Single payer'),
        'description': 'Views on healthcare systems'
    },
    'social_media_regulation': {
        'scale': (-100, 100),
        'labels': ('Free speech absolutist', 'Heavy moderation'),
        'description': 'Views on social media content moderation'
    },
    'immigration': {
        'scale': (-100, 100),
        'labels': ('Closed borders', 'Open borders'),
        'description': 'Views on immigration policy'
    },
    'nuclear_energy': {
        'scale': (-100, 100),
        'labels': ('Anti-nuclear', 'Pro-nuclear'),
        'description': 'Views on nuclear energy adoption'
    },
    'education_reform': {
        'scale': (-100, 100),
        'labels': ('Traditional methods', 'Progressive reform'),
        'description': 'Views on education system changes'
    },
    'gun_control': {
        'scale': (-100, 100),
        'labels': ('No restrictions', 'Strict control'),
        'description': 'Views on gun ownership and regulation'
    },
    'cryptocurrency': {
        'scale': (-100, 100),
        'labels': ('Crypto skeptic', 'Crypto advocate'),
        'description': 'Views on cryptocurrency adoption'
    },
    'space_exploration': {
        'scale': (-100, 100),
        'labels': ('Focus on Earth', 'Expand to space'),
        'description': 'Views on space exploration investment'
    },
    'genetic_engineering': {
        'scale': (-100, 100),
        'labels': ('Ban modification', 'Embrace enhancement'),
        'description': 'Views on human genetic modification'
    },
    'urban_development': {
        'scale': (-100, 100),
        'labels': ('Preserve character', 'Dense development'),
        'description': 'Views on urban growth and zoning'
    },
    'privacy_vs_security': {
        'scale': (-100, 100),
        'labels': ('Total privacy', 'Security priority'),
        'description': 'Views on privacy versus security trade-offs'
    }
}

# Simulation Parameters
DEFAULT_SIMULATION_PARAMS = {
    'interaction_probability': 0.15,
    'homophily_bias': 0.6,
    'max_interactions_per_agent': 2,
    'rounds': 20
}

# Agent Generation Parameters
AGENT_GENERATION_PARAMS = {
    'min_age': 18,
    'max_age': 80,
    'education_distribution': {
        'no_high_school': 0.1,
        'high_school': 0.3,
        'some_college': 0.15,
        'associates': 0.1,
        'bachelors': 0.25,
        'masters': 0.08,
        'phd': 0.02
    },
    'occupation_categories': {
        'unemployed': ['Unemployed', 'Between jobs', 'Job searching'],
        'service': ['Cashier', 'Retail worker', 'Food service worker', 'Janitor', 
                   'Security guard', 'Home health aide', 'Delivery driver'],
        'manual_labor': ['Construction worker', 'Factory worker', 'Warehouse worker',
                        'Landscaper', 'Mechanic', 'Plumber', 'Electrician'],
        'clerical': ['Administrative assistant', 'Data entry clerk', 'Receptionist',
                    'Bank teller', 'Call center agent', 'Office clerk'],
        'skilled_trades': ['Carpenter', 'Welder', 'HVAC technician', 'Auto mechanic',
                          'Hair stylist', 'Chef', 'Dental hygienist'],
        'education': ['Teacher', 'Teaching assistant', 'School counselor', 
                     'Librarian', 'Tutor', 'Daycare worker'],
        'healthcare': ['Nurse', 'Medical assistant', 'EMT', 'Physical therapist',
                      'Pharmacy technician', 'Lab technician'],
        'business': ['Sales representative', 'Manager', 'Accountant', 'HR specialist',
                    'Marketing coordinator', 'Real estate agent'],
        'professional': ['Software engineer', 'Doctor', 'Lawyer', 'Architect',
                        'Financial analyst', 'Consultant', 'Research scientist'],
        'creative': ['Graphic designer', 'Writer', 'Musician', 'Artist',
                    'Photographer', 'Web designer', 'Content creator'],
        'public_service': ['Police officer', 'Firefighter', 'Social worker',
                          'Postal worker', 'Government clerk', 'Military service']
    },
    'socioeconomic_distribution': {
        'poverty': 0.12,
        'working_poor': 0.18,
        'working_class': 0.30,
        'middle_class': 0.25,
        'upper_middle_class': 0.12,
        'wealthy': 0.03
    },
    'personality_biases': [
        'high openness',
        'high conscientiousness',
        'high agreeableness',
        'high extraversion',
        'high emotionality',
        'low agreeableness',
        'balanced'
    ]
}

# Interaction Planning Parameters
INTERACTION_PARAMS = {
    'max_topics_per_conversation': 3,
    'min_topic_score': 20,
    'disagreement_bonus': 1.2,  # Multiplier for topics with disagreement
    'low_knowledge_penalty': 0.7  # Multiplier for topics with low knowledge
}

# State Update Parameters
STATE_UPDATE_PARAMS = {
    'max_position_change_per_conversation': 0.15,  # 15% max change
    'certainty_increase_threshold': 0.7,  # Argument quality needed to increase certainty
    'emotional_reactivity_multiplier': 1.5,  # For high emotionality agents
    'stress_tolerance_impact': 1.3  # Impact multiplier for low stress tolerance
}

# Analysis Parameters
ANALYSIS_PARAMS = {
    'polarization_gap_threshold': 30,  # Gap between opinions to consider polarized
    'echo_chamber_similarity_threshold': 0.7,  # Similarity needed for echo chamber
    'echo_chamber_min_interactions': 3,  # Minimum interactions to form echo chamber
    'influencer_top_n': 10  # Number of top influencers to identify
}

# File I/O Settings
OUTPUT_SETTINGS = {
    'checkpoint_frequency': 5,  # Save checkpoint every N rounds
    'checkpoint_dir': 'checkpoints',
    'results_dir': 'results',
    'log_conversations': True,
    'log_state_changes': True
}

# Visualization Settings
VISUALIZATION_SETTINGS = {
    'figure_size': (12, 8),
    'color_scheme': 'viridis',
    'show_agent_names': True,
    'animation_interval': 500  # milliseconds
}

# Logging Configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'detailed': {
            'format': '%(asctime)s [%(levelname)s] %(name)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'simple': {
            'format': '%(levelname)s - %(message)s'
        },
        'json': {
            'format': '{"time": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'simple',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.FileHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'filename': 'simulation.log',
            'mode': 'a'
        },
        'rotating_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'filename': 'simulation_rotating.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5
        }
    },
    'loggers': {
        'dynavox': {
            'level': 'DEBUG',
            'handlers': ['console', 'file'],
            'propagate': False
        },
        'dynavox.llm': {
            'level': 'INFO',
            'handlers': ['console', 'rotating_file'],
            'propagate': False
        },
        'dynavox.simulation': {
            'level': 'DEBUG',
            'handlers': ['console', 'file'],
            'propagate': False
        },
        'dynavox.interactions': {
            'level': 'DEBUG',
            'handlers': ['console', 'file'],
            'propagate': False
        },
        'dynavox.agents': {
            'level': 'INFO',
            'handlers': ['console', 'file'],
            'propagate': False
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console']
    }
}

# Logging Levels
LOG_LEVELS = {
    'DEBUG': 'Detailed information for diagnosing problems',
    'INFO': 'General informational messages',
    'WARNING': 'Warning messages about potential issues',
    'ERROR': 'Error messages when something goes wrong',
    'CRITICAL': 'Critical messages for severe failures'
}

# Default log level (can be overridden by environment variable)
DEFAULT_LOG_LEVEL = 'INFO'