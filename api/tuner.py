# api/tuner.py - Core tuning engine

import itertools
import random
import time
import logging
import pandas as pd
from typing import Dict, List, Any, Union
from django.utils import timezone
from .models import TuningJob, TuningRun
from .engine import HybridBacktester, UniversalStrategyParser



logger = logging.getLogger(__name__)


class TuningEngine:
    """Core engine for parameter optimization using grid or random search"""
    
    def __init__(self):
        self.backtester = None
    
    def generate_parameter_combinations(self, param_space: Dict, search_type: str, budget: int) -> List[Dict]:
        """
        Generate parameter combinations based on search type and budget
        
        Args:
            param_space: Dictionary defining parameter ranges
            search_type: 'grid' or 'random'
            budget: Maximum number of combinations to generate
            
        Returns:
            List of parameter dictionaries
        """
        combinations = []
        
        if search_type == 'grid':
            combinations = self._generate_grid_combinations(param_space)
            # Limit to budget if grid is too large
            if len(combinations) > budget:
                combinations = combinations[:budget]
                
        elif search_type == 'random':
            combinations = self._generate_random_combinations(param_space, budget)
            
        return combinations
    
    def _generate_grid_combinations(self, param_space: Dict) -> List[Dict]:
        """Generate all possible parameter combinations for grid search"""
        param_names = list(param_space.keys())
        param_values = []
        
        for param_name, param_config in param_space.items():
            if param_config['type'] == 'int':
                values = list(range(param_config['min'], param_config['max'] + 1))
            elif param_config['type'] == 'float':
                # For float ranges, create a reasonable number of steps
                steps = param_config.get('steps', 10)
                min_val, max_val = param_config['min'], param_config['max']
                step_size = (max_val - min_val) / (steps - 1)
                values = [round(min_val + i * step_size, 4) for i in range(steps)]
            elif param_config['type'] == 'choice':
                values = param_config['choices']
            else:
                raise ValueError(f"Unsupported parameter type: {param_config['type']}")
            
            param_values.append(values)
        
        # Generate all combinations
        combinations = []
        for combo in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combo))
            combinations.append(param_dict)
            
        return combinations
    
    def _generate_random_combinations(self, param_space: Dict, budget: int) -> List[Dict]:
        """Generate random parameter combinations"""
        combinations = []
        
        for _ in range(budget):
            param_dict = {}
            
            for param_name, param_config in param_space.items():
                if param_config['type'] == 'int':
                    value = random.randint(param_config['min'], param_config['max'])
                elif param_config['type'] == 'float':
                    value = round(random.uniform(param_config['min'], param_config['max']), 4)
                elif param_config['type'] == 'choice':
                    value = random.choice(param_config['choices'])
                else:
                    raise ValueError(f"Unsupported parameter type: {param_config['type']}")
                
                param_dict[param_name] = value
            
            combinations.append(param_dict)
        
        return combinations
    
    # api/tuner.py
# ... (rest of the file remains the same)

    def run_tuner(self, job_id: str):
        """Main method to run the parameter tuning process"""
        try:
            job = TuningJob.objects.get(job_id=job_id)
            job.status = 'running'
            job.save()

            # Load data once for the entire job
            data = pd.read_csv('aapl_data.csv') #

            combinations = self.generate_parameter_combinations(
                job.param_space, job.search_type, job.budget
            )

            job.total_runs = len(combinations)
            job.save()

            logger.info(f"Starting tuning job {job_id} with {len(combinations)} combinations")

            self.backtester = HybridBacktester()

            for i, params in enumerate(combinations):
                try:
                    # Pass the data to the single backtest runner
                    self._run_single_backtest(job, params, data)

                    job.progress = i + 1
                    job.save()

                    logger.info(f"Completed run {i+1}/{len(combinations)} for job {job_id}")

                except Exception as e:
                    logger.error(f"Error in run {i+1} for job {job_id}: {str(e)}")
                    continue

            job.status = 'completed'
            job.completed_at = timezone.now()
            job.save()

            logger.info(f"Tuning job {job_id} completed successfully")

        except Exception as e:
            logger.error(f"Error in tuning job {job_id}: {str(e)}")
            job = TuningJob.objects.get(job_id=job_id)
            job.status = 'failed'
            job.error_message = str(e)
            job.save()
            raise

    def _run_single_backtest(self, job: TuningJob, params: Dict, data: pd.DataFrame):
        """Run a single backtest with given parameters and data"""
        start_time = time.time()

        try:
            # Create a full strategy config by merging the base spec and the parameters
            strategy_config = job.strategy_spec.copy()

            # Assuming params are top-level and need to be merged into a specific block,
            # or directly passed. Let's assume for now they update the strategy_config.
            # You might need to adjust this based on your front-end schema.
            # Example: strategy_config['params'].update(params)

            # For simplicity, let's assume the params are for indicators and
            # need to be applied directly to the nodes of the strategy_spec
            updated_strategy_spec = self._apply_params_to_strategy(strategy_config, params)

            # Parse the updated strategy
            parser = UniversalStrategyParser(updated_strategy_spec)
            parsed_strategy = parser.parse()

            # Now run the backtest with the parsed strategy and data
            results = self.backtester.run_backtest(data, parsed_strategy)

            execution_time = time.time() - start_time

            TuningRun.objects.create(
                job=job,
                params=params,
                metrics_json=results.get('metrics', {}),
                equity_curve_data=results.get('equity_curve', []),
                trades_data=results.get('trades', []),
                execution_time=execution_time
            )

        except Exception as e:
            TuningRun.objects.create(
                job=job,
                params=params,
                metrics_json={'error': str(e)},
                equity_curve_data=[],
                trades_data=[],
                execution_time=time.time() - start_time
            )
            raise

    def _apply_params_to_strategy(self, strategy_spec: Dict, params: Dict) -> Dict:
        """Helper to apply parameters to the strategy spec, this is a placeholder."""
        # This function would need to be implemented to correctly
        # find and update the right nodes in the strategy_spec's
        # 'nodes' list.
        # For an MVP, you might have a simpler strategy format where
        # params are at the top level and this step is not needed.
        return strategy_spec

    # ... (rest of the file remains the same)
    
    def get_best_results(self, job_id: str, top_n: int = 10, objective: str = None) -> List[Dict]:
        """
        Get the best results from a completed tuning job
        
        Args:
            job_id: ID of the tuning job
            top_n: Number of top results to return
            objective: Metric to sort by (defaults to job's objective)
            
        Returns:
            List of best results with parameters and metrics
        """
        job = TuningJob.objects.get(job_id=job_id)
        objective = objective or job.objective
        
        # Get all successful runs
        runs = TuningRun.objects.filter(
            job=job,
            metrics_json__has_key=objective
        ).exclude(
            metrics_json__has_key='error'
        )
        
        # Sort by objective metric (assuming higher is better for most metrics)
        if objective in ['max_drawdown_pct']:
            # For drawdown, lower is better
            runs = runs.order_by(f'metrics_json__{objective}')
        else:
            # For most metrics, higher is better
            runs = runs.extra(
                select={f'objective_value': f"CAST(json_extract(metrics_json, '$.{objective}') AS REAL)"}
            ).order_by('-objective_value')
        
        # Get top N results
        top_runs = runs[:top_n]
        
        results = []
        for run in top_runs:
            results.append({
                'run_id': run.run_id,
                'params': run.params,
                'metrics': run.metrics_json,
                'objective_value': run.metrics_json.get(objective, 0),
                'execution_time': run.execution_time
            })
        
        return results