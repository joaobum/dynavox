"""Quick and easy simulation runner for OpinionDynamics."""
import os
import sys
import logging
from typing import Optional, List, Dict
from dotenv import load_dotenv

# Add parent directory to path if running as script
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm import create_llm_client
from src.simulation import SimulationEngine
from src.analysis import SimulationVisualizer, SimulationReporter
from src.config import STANDARD_TOPICS
from src.utils.enhanced_logging import setup_enhanced_logging, RealTimeDataWriter

# Load environment variables
load_dotenv()

# Logging will be initialized by setup_enhanced_logging
logger = logging.getLogger('dynavox')


class QuickSimulation:
    """Simplified interface for running OpinionDynamics simulations."""
    
    def __init__(self, model: Optional[str] = None, use_mock: bool = False, log_level: str = 'INFO', use_async: bool = False):
        """Initialize simulation with automatic configuration.
        
        Args:
            model: Model name (e.g., 'gpt-4o', 'claude-3-5-sonnet')
                  If None, uses DEFAULT_MODEL from .env
            use_mock: Use mock LLM for testing (overrides model)
            log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
            use_async: Use async execution for parallel conversations
        """
        self.log_level = log_level
        self.use_mock = use_mock
        self.use_async = use_async
        # Set log level
        numeric_level = getattr(logging, log_level.upper())
        for logger_name in ['dynavox', 'dynavox.simulation', 
                           'dynavox.llm', 'dynavox.interactions', 
                           'dynavox.agents']:
            logging.getLogger(logger_name).setLevel(numeric_level)
        logger.info(f"Log level set to {log_level}")
        
        if use_mock:
            os.environ['USE_MOCK_LLM'] = 'true'
            logger.info("Using Mock LLM for testing")
        
        # Create LLM client with automatic configuration
        self.llm_client = create_llm_client(model)
        
        # Initialize simulation engine with async support
        self.sim = SimulationEngine(self.llm_client, use_async=use_async)
        
        # Initialize analysis tools
        self.visualizer = None
        self.reporter = None
    
    def run(self, 
            num_agents: int = 20,
            num_rounds: int = 10,
            topics: Optional[List[str]] = None,
            output_dir: str = "results",
            simulation_name: str = "test",
            **kwargs) -> Dict:
        """Run a complete simulation with analysis.
        
        Args:
            num_agents: Number of agents to create
            num_rounds: Number of simulation rounds
            topics: List of topics (uses defaults if None)
            output_dir: Directory for all outputs
            **kwargs: Additional arguments for run_simulation
        
        Returns:
            Dictionary with paths to generated outputs
        """
        # Use default topics if none provided
        if topics is None:
            topics = ['climate_change', 'wealth_inequality', 'ai_regulation']
        
        # Extract simulation parameters
        interaction_prob = kwargs.get('interaction_probability', 0.15)
        homophily_bias = kwargs.get('homophily_bias', 0.6)
        model_name = getattr(self.llm_client, 'model', 'Unknown')
        
        # Set up enhanced logging and output directory
        sim_output_dir, log_path = setup_enhanced_logging(
            output_dir=output_dir,
            model_name=model_name,
            num_agents=num_agents,
            num_rounds=num_rounds,
            interaction_prob=interaction_prob,
            homophily=homophily_bias,
            is_mock=self.use_mock,
            log_level=self.log_level
        )
        
        # Initialize real-time data writer
        self.data_writer = RealTimeDataWriter(os.path.join(sim_output_dir, 'data'))
        
        # Attach data writer to simulation engine
        self.sim.data_writer = self.data_writer
        
        logger.info("=== Starting OpinionDynamics Simulation ===")
        logger.info(f"Model: {getattr(self.llm_client, 'model', 'Unknown')}")
        logger.info(f"Agents: {num_agents}")
        logger.info(f"Rounds: {num_rounds}")
        logger.info(f"Topics: {', '.join(topics)}")
        logger.info(f"Output: {sim_output_dir}")
        
        # Initialize population
        logger.info("Creating agent population...")
        self.sim.initialize_population(
            size=num_agents,
            topics=topics
        )
        
        # Run simulation
        logger.info("Running simulation...")
        self.sim.run_simulation(
            rounds=num_rounds,
            interaction_probability=kwargs.get('interaction_probability', 0.15),
            homophily_bias=kwargs.get('homophily_bias', 0.6),
            max_interactions_per_agent=kwargs.get('max_interactions_per_agent', 2)
        )
        
        # Export raw results to data subdirectory
        logger.info("Exporting results...")
        self.sim.export_results(sim_output_dir)
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        viz_dir = os.path.join(sim_output_dir, "visualizations")
        self.visualizer = SimulationVisualizer(self.sim, viz_dir)
        self.visualizer.create_all_visualizations()
        self.visualizer.create_summary_dashboard()
        
        # Generate reports
        logger.info("Generating reports...")
        report_dir = os.path.join(sim_output_dir, "reports")
        self.reporter = SimulationReporter(self.sim, report_dir)
        report_path = self.reporter.generate_full_report()
        self.reporter.generate_csv_exports()
        self.reporter.generate_json_summary()
        
        # Save checkpoints if any were created during simulation
        checkpoint_dir = os.path.join(sim_output_dir, "checkpoints")
        if hasattr(self.sim, '_checkpoint_dir'):
            import shutil
            if os.path.exists(self.sim._checkpoint_dir):
                shutil.move(self.sim._checkpoint_dir, checkpoint_dir)
        
        logger.info("=== Simulation Complete ===")
        logger.info(f"All outputs saved to: {sim_output_dir}/")
        
        # Return paths to key outputs
        return {
            'output_dir': sim_output_dir,
            'data': os.path.join(sim_output_dir, 'data'),
            'visualizations': viz_dir,
            'reports': report_dir,
            'logs': os.path.join(sim_output_dir, 'logs'),
            'log_file': log_path,
            'checkpoints': checkpoint_dir if os.path.exists(checkpoint_dir) else None,
            'summary_dashboard': os.path.join(viz_dir, 'summary_dashboard.png'),
            'full_report': report_path,
            'simulation': self.sim
        }
    
    def quick_run(self, preset: str = "small") -> Dict:
        """Run simulation with preset configurations.
        
        Args:
            preset: Configuration preset
                   - 'test': 5 agents, 3 rounds (quick test)
                   - 'small': 20 agents, 10 rounds (default)
                   - 'medium': 50 agents, 20 rounds
                   - 'large': 100 agents, 30 rounds
        
        Returns:
            Dictionary with paths to generated outputs
        """
        presets = {
            'test': {'num_agents': 5, 'num_rounds': 3, 'simulation_name': 'test'},
            'small': {'num_agents': 20, 'num_rounds': 10, 'simulation_name': 'small'},
            'medium': {'num_agents': 50, 'num_rounds': 20, 'simulation_name': 'medium'},
            'large': {'num_agents': 100, 'num_rounds': 30, 'simulation_name': 'large'}
        }
        
        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Choose from: {list(presets.keys())}")
        
        config = presets[preset]
        logger.info(f"Running '{preset}' simulation preset...")
        
        return self.run(**config)


def main():
    """Command-line interface for quick simulations."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run OpinionDynamics simulation')
    parser.add_argument('--model', type=str, default=None,
                       help='LLM model to use (e.g., gpt-4o, claude-3-5-sonnet)')
    parser.add_argument('--mock', action='store_true',
                       help='Use mock LLM for testing')
    parser.add_argument('--preset', type=str, default='small',
                       choices=['test', 'small', 'medium', 'large'],
                       help='Simulation size preset')
    parser.add_argument('--agents', type=int, default=None,
                       help='Number of agents (overrides preset)')
    parser.add_argument('--rounds', type=int, default=None,
                       help='Number of rounds (overrides preset)')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--name', type=str, default=None,
                       help='Simulation name (for output folder)')
    
    args = parser.parse_args()
    
    # Create simulation
    sim = QuickSimulation(model=args.model, use_mock=args.mock)
    
    # Run with custom parameters or preset
    if args.agents is not None or args.rounds is not None:
        # Custom configuration
        config = {
            'num_agents': args.agents or 20,
            'num_rounds': args.rounds or 10,
            'output_dir': args.output,
            'simulation_name': args.name or 'custom'
        }
        results = sim.run(**config)
    else:
        # Use preset
        results = sim.quick_run(preset=args.preset)
    
    # Print summary
    logger.info("=" * 50)
    logger.info("SIMULATION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Dashboard: {results['summary_dashboard']}")
    print(f"Full Report: {results['full_report']}")
    print(f"All Results: {results['output_dir']}/")
    print("\nTo view the dashboard image:")
    print(f"  open {results['summary_dashboard']}")


if __name__ == "__main__":
    main()