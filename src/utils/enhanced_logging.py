"""Enhanced logging configuration for DynaVox with unicode indicators and real-time updates."""
import os
import sys
import logging
import logging.handlers
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
import json


class UnicodeConsoleHandler(logging.StreamHandler):
    """Console handler that adds unicode prefixes to log messages."""
    
    UNICODE_PREFIXES = {
        'DEBUG': '🐛',
        'INFO': 'ℹ️',
        'WARNING': '⚠️',
        'ERROR': '❌',
        'CRITICAL': '🔥',
        # Custom prefixes for specific message types
        'AGENT_CREATE': '👤',
        'CONVERSATION': '💬',
        'OPINION_CHANGE': '🔄',
        'EMOTION_CHANGE': '😊',
        'ANALYSIS': '📊',
        'SAVE': '💾',
        'COMPLETE': '✅',
        'START': '🚀',
        'ROUND': '🔁',
        'THINK': '🤔',
        'INTERACT': '🤝',
        'PARALLEL': '⚡',
        'CONFIG': '⚙️',
    }
    
    def emit(self, record):
        """Emit a record with unicode prefix."""
        try:
            # Skip DEBUG messages if handler level is INFO or higher
            if record.levelno < self.level:
                return
                
            msg = self.format(record)
            
            # Check for specific keywords and add appropriate unicode
            for keyword, prefix in self.UNICODE_PREFIXES.items():
                if keyword.lower() in msg.lower():
                    msg = f"{prefix} {msg}"
                    break
            else:
                # Default prefix based on level
                prefix = self.UNICODE_PREFIXES.get(record.levelname, '')
                if prefix:
                    msg = f"{prefix} {msg}"
            
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


class SimulationFileHandler(logging.FileHandler):
    """Enhanced file handler that captures all console output."""
    
    def __init__(self, filename, mode='w', encoding='utf-8', delay=False):
        super().__init__(filename, mode, encoding, delay)
        # Use a more detailed formatter for files
        self.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)-8s] %(name)-25s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))


def setup_enhanced_logging(output_dir: str, model_name: str, num_agents: int,
                          num_rounds: int, interaction_prob: float, homophily: float,
                          is_mock: bool = False, log_level: str = 'INFO') -> Tuple[str, str]:
    """Set up enhanced simulation logging with detailed naming.
    
    Args:
        output_dir: Base output directory
        model_name: Name of the LLM model
        num_agents: Number of agents
        num_rounds: Number of rounds
        interaction_prob: Interaction probability (0-1)
        homophily: Homophily bias (0-1)
        is_mock: Whether using mock LLM
        log_level: Logging level
        
    Returns:
        Tuple of (full output directory path, log file path)
    """
    # Create timestamp without seconds
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    
    # Clean model name for filesystem
    model_for_filename = model_name.replace(':', '-').replace('/', '-')
    if is_mock:
        model_for_filename = 'mock'
    
    # Format interaction probability and homophily as percentages
    interaction_pct = int(interaction_prob * 100)
    homophily_pct = int(homophily * 100)
    
    # Create directory name with all parameters
    dir_name = f"{timestamp}_{model_for_filename}_{num_agents}_{num_rounds}_{interaction_pct}_{homophily_pct}"
    
    full_output_dir = os.path.join(output_dir, dir_name)
    os.makedirs(full_output_dir, exist_ok=True)
    
    # Create logs directory
    logs_dir = os.path.join(full_output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create data directory for real-time updates
    data_dir = os.path.join(full_output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Log filename matches directory name
    log_filename = f"{dir_name}.log"
    log_path = os.path.join(logs_dir, log_filename)
    
    # Remove existing handlers to avoid duplicates
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set up console handler with unicode
    console_handler = UnicodeConsoleHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Set up file handler
    file_handler = SimulationFileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)  # Capture everything in file
    
    # Configure root logger
    # Set root level to the minimum of console and file levels to ensure all messages are captured
    root_logger.setLevel(min(getattr(logging, log_level.upper()), logging.DEBUG))
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Configure all dynavox loggers
    for logger_name in ['dynavox', 'dynavox.simulation', 'dynavox.llm', 
                       'dynavox.interactions', 'dynavox.agents', 'dynavox.analysis']:
        logger = logging.getLogger(logger_name)
        # Set logger level to match requested log level, not always DEBUG
        logger.setLevel(getattr(logging, log_level.upper()))
        logger.propagate = True
    
    # Suppress HTTP logging
    for lib in ['httpx', 'openai', 'anthropic', 'httpcore', 'urllib3', 'requests']:
        logging.getLogger(lib).setLevel(logging.WARNING)
    
    # Log initial configuration
    logger = logging.getLogger('dynavox')
    logger.info(f"🚀 === DynaVox Simulation Starting ===")
    logger.info(f"⚙️ Configuration:")
    logger.info(f"  📁 Output Directory: {full_output_dir}")
    logger.info(f"  🤖 Model: {model_name} {'(Mock)' if is_mock else ''}")
    logger.info(f"  👥 Agents: {num_agents}")
    logger.info(f"  🔁 Rounds: {num_rounds}")
    logger.info(f"  🤝 Interaction Probability: {interaction_prob:.2%}")
    logger.info(f"  🧲 Homophily Bias: {homophily:.2%}")
    logger.info(f"  📝 Log Level: {log_level}")
    logger.info(f"  📄 Log File: {log_path}")
    
    return full_output_dir, log_path


class RealTimeDataWriter:
    """Handles real-time writing of simulation data to files."""
    
    def __init__(self, data_dir: str):
        """Initialize with data directory."""
        self.data_dir = data_dir
        self.logger = logging.getLogger('dynavox.data')
        
        # Initialize files
        self._init_files()
    
    def _init_files(self):
        """Initialize empty data files."""
        # Create empty files that will be updated
        self.agents_file = os.path.join(self.data_dir, 'agents_realtime.json')
        self.conversations_file = os.path.join(self.data_dir, 'conversations_realtime.json')
        self.metrics_file = os.path.join(self.data_dir, 'metrics_history_realtime.json')
        
        # Initialize with empty structures
        self._write_json(self.agents_file, {})
        self._write_json(self.conversations_file, [])
        self._write_json(self.metrics_file, [])
        
        self.logger.info(f"💾 Initialized real-time data files in {self.data_dir}")
    
    def _write_json(self, filepath: str, data: Any):
        """Write JSON data to file."""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def update_agent(self, agent):
        """Update agent data in real-time."""
        try:
            # Read current agents
            with open(self.agents_file, 'r') as f:
                agents = json.load(f)
            
            # Update agent data
            agents[agent.id] = {
                'id': agent.id,
                'name': agent.name,
                'age': agent.background.age,
                'occupation': agent.background.occupation,
                'personality': agent.personality.__dict__,
                'opinions': {k: v.__dict__ for k, v in agent.opinions.items()},
                'emotional_state': agent.emotional_state.__dict__,
                'last_updated': datetime.now().isoformat()
            }
            
            # Write back
            self._write_json(self.agents_file, agents)
            self.logger.debug(f"💾 Updated agent {agent.name} in real-time")
            
        except Exception as e:
            self.logger.error(f"Failed to update agent data: {e}")
    
    def add_conversation(self, conversation):
        """Add conversation data in real-time."""
        try:
            # Read current conversations
            with open(self.conversations_file, 'r') as f:
                conversations = json.load(f)
            
            # Add new conversation
            conv_data = {
                'id': len(conversations),
                'participants': conversation.participants,
                'timestamp': conversation.timestamp.isoformat(),
                'duration_turns': conversation.duration_turns,
                'topics': conversation.plan.topics,
                'summary': getattr(conversation, 'summary', 'No summary'),
                'state_changes': {
                    pid: {
                        'opinion_changes': changes.opinion_changes,
                        'emotion_changes': changes.emotion_changes
                    }
                    for pid, changes in conversation.state_changes.items()
                }
            }
            
            conversations.append(conv_data)
            
            # Write back
            self._write_json(self.conversations_file, conversations)
            self.logger.debug(f"💾 Added conversation #{len(conversations)} in real-time")
            
        except Exception as e:
            self.logger.error(f"Failed to update conversation data: {e}")
    
    def update_metrics(self, metrics):
        """Update metrics history in real-time."""
        try:
            # Read current metrics
            with open(self.metrics_file, 'r') as f:
                metrics_history = json.load(f)
            
            # Add new metrics
            metrics_data = {
                'round': metrics.round_number,
                'timestamp': metrics.timestamp.isoformat(),
                'overall_polarization': metrics.overall_polarization,
                'overall_consensus': metrics.overall_consensus,
                'avg_certainty': metrics.avg_certainty,
                'avg_emotional_valence': metrics.avg_emotional_valence,
                'interaction_count': metrics.interaction_count,
                'opinion_metrics': metrics.opinion_metrics
            }
            
            metrics_history.append(metrics_data)
            
            # Write back
            self._write_json(self.metrics_file, metrics_history)
            self.logger.debug(f"💾 Updated metrics for round {metrics.round_number} in real-time")
            
        except Exception as e:
            self.logger.error(f"Failed to update metrics data: {e}")


# Convenience function to replace the old one
def get_simulation_output_dir(base_dir: str, simulation_name: str, 
                            is_mock: bool = False) -> str:
    """Legacy function for compatibility."""
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    mock_suffix = "_mock" if is_mock else ""
    dir_name = f"{timestamp}_{simulation_name}{mock_suffix}"
    
    output_dir = os.path.join(base_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir