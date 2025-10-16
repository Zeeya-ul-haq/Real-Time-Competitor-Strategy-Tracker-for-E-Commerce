#!/usr/bin/env python3
"""
E-Commerce Competitor Analysis System - Main Orchestrator
Professional implementation with configuration-driven architecture
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Callable, Tuple
from datetime import datetime
import subprocess
import importlib.util
from dataclasses import dataclass

# Import configuration manager
from config_manager import get_config_manager, BaseComponent

@dataclass 
class FlowResult:
    """Data class for flow execution results"""
    name: str
    success: bool
    error_message: str = ""
    execution_time: float = 0.0

class SystemOrchestrator(BaseComponent):
    """
    Professional system orchestrator with configuration-driven flow management
    """

    def __init__(self):
        super().__init__("orchestrator")
        self.flow_registry: Dict[str, Callable[[], bool]] = {}
        self.execution_results: List[FlowResult] = []

    def register_flows(self) -> None:
        """Register all available flows with their execution functions"""
        self.flow_registry = {
            'amazon_scraper': self._run_amazon_scraper,
            'flipkart_scraper': self._run_flipkart_scraper,
            'data_processing': self._run_data_processing,
            'ml_training': self._run_ml_training,
            'dashboard': self._run_dashboard,
            'rag_system': self._run_rag_system
        }

    def _import_module(self, module_name: str, file_path: str) -> Any:
        """Dynamically import a module from file path"""
        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load module {module_name} from {file_path}")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            self.logger.error(f"Failed to import {module_name}: {e}")
            raise

    def _execute_flow(self, flow_name: str, flow_function: Callable[[], bool]) -> FlowResult:
        """Execute a single flow with error handling and timing"""
        start_time = datetime.now()

        try:
            self.logger.info(f"Starting {flow_name}...")
            success = flow_function()

            execution_time = (datetime.now() - start_time).total_seconds()

            if success:
                self.logger.info(f" {flow_name} completed successfully in {execution_time:.2f}s")
                return FlowResult(flow_name, True, "", execution_time)
            else:
                self.logger.error(f" {flow_name} failed")
                return FlowResult(flow_name, False, "Flow returned False", execution_time)

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = str(e)
            self.logger.error(f" {flow_name} failed with exception: {error_msg}")
            return FlowResult(flow_name, False, error_msg, execution_time)

    def _get_user_confirmation(self, flow_name: str, description: str) -> bool:
        """Get user confirmation to run a flow"""
        while True:
            try:
                response = input(f"\n Run {flow_name}?\n   {description}\n   [y/N]: ").lower().strip()
                if response in ['y', 'yes']:
                    return True
                elif response in ['n', 'no', '']:
                    return False
                else:
                    print("   Please enter 'y' for yes or 'n' for no")
            except KeyboardInterrupt:
                print("\n Process interrupted by user")
                sys.exit(0)

    def _run_amazon_scraper(self) -> bool:
        """Execute Amazon scraper flow"""
        try:
            scraper_module = self._import_module("amazon_scraper", "amazon_scraper.py")
            scraper_config = self.config_manager.get_scraper_config('amazon')
            return scraper_module.run_scraper(scraper_config, self.paths)
        except Exception as e:
            self.logger.error(f"Amazon scraper execution failed: {e}")
            return False

    def _run_flipkart_scraper(self) -> bool:
        """Execute Flipkart scraper flow"""
        try:
            scraper_module = self._import_module("flipkart_scraper", "flipkart_scraper.py")
            scraper_config = self.config_manager.get_scraper_config('flipkart')
            return scraper_module.run_scraper(scraper_config, self.paths)
        except Exception as e:
            self.logger.error(f"Flipkart scraper execution failed: {e}")
            return False

    def _run_data_processing(self) -> bool:
        """Execute data processing flow"""
        try:
            processing_module = self._import_module("data_processor", "data_processor.py")
            processing_config = self.config.get('data_processing')
            return processing_module.run_processing(processing_config, self.paths)
        except Exception as e:
            self.logger.error(f"Data processing execution failed: {e}")
            return False

    def _run_ml_training(self) -> bool:
        """Execute ML training flow"""
        try:
            ml_module = self._import_module("ml_trainer", "ml_trainer.py")
            ml_config = self.config.get('machine_learning')
            return ml_module.run_training(ml_config, self.paths)
        except Exception as e:
            self.logger.error(f"ML training execution failed: {e}")
            return False

    def _run_dashboard(self) -> bool:
        """Execute dashboard flow"""
        try:
            dashboard_config = self.config.get('dashboard', {})
            server_config = dashboard_config.get('server', {})

            # Auto-train ML models if configured
            if (self.config.get('machine_learning', {}).get('training', {}).get('auto_train_before_dashboard', False) and 
                self.config_manager.is_flow_enabled('ml_training')):
                self.logger.info("Auto-training ML models before dashboard...")
                self._run_ml_training()

            # Launch Streamlit dashboard
            host = server_config.get('host', 'localhost')
            port = server_config.get('port', 8501)

            cmd = [
                sys.executable, "-m", "streamlit", "run", "dashboard.py",
                "--server.address", host,
                "--server.port", str(port),
                "--server.headless", "true",
                "--server.runOnSave", "true"
            ]

            subprocess.run(cmd, check=True)
            return True

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Dashboard subprocess failed: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Dashboard execution failed: {e}")
            return False

    def _run_rag_system(self) -> bool:
        """Execute RAG system flow"""
        try:
            rag_module = self._import_module("rag_system", "rag_system.py")
            rag_config = self.config.get('rag_system')
            return rag_module.run_rag(rag_config, self.paths)
        except Exception as e:
            self.logger.error(f"RAG system execution failed: {e}")
            return False

    def print_banner(self) -> None:
        """Print professional system banner"""
        project_config = self.config.get('project', {})
        project_name = project_config.get('name', 'E-Commerce Analysis System')
        version = project_config.get('version', '1.0.0')

        banner = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║    🚀 {project_name:<30} v{version:<10}                     ║
║                                                                              ║
║    Professional E-Commerce Intelligence Platform                            ║
║    Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<20}                                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
        print(banner)

    def print_execution_summary(self) -> None:
        """Print comprehensive execution summary"""
        print("\n" + "="*80)
        print(" EXECUTION SUMMARY")
        print("="*80)

        successful_flows = []
        failed_flows = []
        total_time = 0.0

        for result in self.execution_results:
            total_time += result.execution_time
            status = " SUCCESS" if result.success else " FAILED"
            time_str = f"{result.execution_time:.2f}s"

            print(f"{result.name:<25} | {status:<12} | {time_str:>8}")

            if result.success:
                successful_flows.append(result.name)
            else:
                failed_flows.append(result.name)
                if result.error_message:
                    print(f"{'':27} Error: {result.error_message[:50]}...")

        print("="*80)
        print(f"Total Execution Time: {total_time:.2f}s")
        print(f"Successful Flows: {len(successful_flows)}/{len(self.execution_results)}")

        if failed_flows:
            print(f"\n  Failed Flows: {', '.join(failed_flows)}")
            print("Check logs for detailed error information.")

        print("="*80)

    def get_enabled_flows(self) -> List[Tuple[str, str, str]]:
        """Get list of enabled flows with metadata"""
        enabled_flows = []
        flows_config = self.config.get('flows', {})

        for flow_name, config in flows_config.items():
            if config.get('enabled', False) and flow_name in self.flow_registry:
                flow_display_name = config.get('name', flow_name)
                flow_description = config.get('description', 'No description available')
                enabled_flows.append((flow_name, flow_display_name, flow_description))

        return enabled_flows

    def run(self) -> bool:
        """Main orchestrator execution method"""
        try:
            self.print_banner()
            self.register_flows()

            # Get enabled flows
            enabled_flows = self.get_enabled_flows()

            if not enabled_flows:
                self.logger.warning("No flows enabled in configuration")
                return False

            self.logger.info(f"Found {len(enabled_flows)} enabled flows")

            # Interactive flow selection and execution
            for flow_name, display_name, description in enabled_flows:
                if self._get_user_confirmation(display_name, description):
                    flow_function = self.flow_registry[flow_name]
                    result = self._execute_flow(display_name, flow_function)
                    self.execution_results.append(result)
                else:
                    self.logger.info(f"Skipped {display_name}")

            # Print summary
            self.print_execution_summary()

            # Keep dashboard running if successful
            dashboard_result = next(
                (r for r in self.execution_results if "Dashboard" in r.name and r.success), 
                None
            )

            if dashboard_result:
                print("\n Dashboard is running... Press Ctrl+C to stop")
                try:
                    while True:
                        pass
                except KeyboardInterrupt:
                    print("\n Shutting down gracefully...")

            return len([r for r in self.execution_results if r.success]) > 0

        except KeyboardInterrupt:
            print("\n Process interrupted by user")
            return False
        except Exception as e:
            self.logger.error(f"Orchestrator execution failed: {e}")
            return False

def main():
    """Entry point for the system orchestrator"""
    try:
        # Initialize configuration manager first
        config_manager = get_config_manager()

        # Create and run orchestrator
        orchestrator = SystemOrchestrator()
        success = orchestrator.run()

        # Exit with appropriate code
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n Process interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f" Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
