"""Enhanced logging configuration for DynaVox."""
import os
import logging
import logging.handlers
from datetime import datetime
from typing import Optional, Dict, Any


def setup_simulation_logging(output_dir: str, simulation_name: str, 
                           is_mock: bool = False, log_level: str = 'INFO') -> str:
    """Set up simulation-specific logging configuration.
    
    Args:
        output_dir: Base output directory for the simulation
        simulation_name: Name of the simulation
        is_mock: Whether this is a mock simulation
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Path to the log file
    """
    # Create logs directory
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    mock_suffix = "_mock" if is_mock else ""
    log_filename = f"{timestamp}_{simulation_name}{mock_suffix}.log"
    
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    log_path = os.path.join(logs_dir, log_filename)
    
    # Configure file handler for simulation
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(getattr(logging, log_level.upper()))
    
    # Use detailed formatter for file logs
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    # Add handler to all dynavox loggers
    for logger_name in ['dynavox', 'dynavox.simulation', 'dynavox.llm', 
                       'dynavox.interactions', 'dynavox.agents']:
        logger = logging.getLogger(logger_name)
        logger.addHandler(file_handler)
    
    # Suppress HTTP logging from external libraries
    for lib in ['httpx', 'openai', 'anthropic', 'httpcore', 'urllib3']:
        logging.getLogger(lib).setLevel(logging.WARNING)
    
    # Log initial setup info
    logger = logging.getLogger('dynavox')
    logger.info(f"Logging initialized for simulation: {simulation_name}")
    logger.info(f"Mock mode: {is_mock}")
    logger.info(f"Log file: {log_path}")
    
    return log_path


def get_simulation_output_dir(base_dir: str, simulation_name: str, 
                            is_mock: bool = False) -> str:
    """Generate simulation output directory with appropriate naming.
    
    Args:
        base_dir: Base directory for outputs
        simulation_name: Name of the simulation
        is_mock: Whether this is a mock simulation
        
    Returns:
        Full path to simulation output directory
    """
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    mock_suffix = "_mock" if is_mock else ""
    dir_name = f"{timestamp}_{simulation_name}{mock_suffix}"
    
    output_dir = os.path.join(base_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir