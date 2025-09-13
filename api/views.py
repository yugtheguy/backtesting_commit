# api/views.py

import json
import pandas as pd
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .engine import UniversalStrategyParser, HybridBacktester, HybridVisualizationEngine
import uuid
import threading
from django.shortcuts import get_object_or_404
from .models import TuningJob, TuningRun
from .tuner import TuningEngine  # Import from tuner.py, don't redefine here!
import logging

logger = logging.getLogger(__name__)

# Define the path to your CSV file
DATA_FILE_PATH = 'aapl_data.csv'

class BacktestAPIView(APIView):
    """
    API endpoint to run a backtest.
    Accepts a POST request with a 'strategy' JSON object.
    """
    def post(self, request, *args, **kwargs):
        try:
            # 1. Get Strategy JSON from request body
            strategy_json = request.data
            if not isinstance(strategy_json, dict):
                strategy_json = json.loads(strategy_json.get('strategy'))

            print(f"Strategy received: {strategy_json.get('name', 'Unnamed')}")

            # 2. Load the historical data from the local file
            try:
                data = pd.read_csv(DATA_FILE_PATH)
                print(f"Data loaded: {len(data)} rows, from {data['Date'].min()} to {data['Date'].max()}")
                
                # Validate required columns
                required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                missing_columns = [col for col in required_columns if col not in data.columns]
                
                if missing_columns:
                    return Response(
                        {"error": f"Missing required columns: {missing_columns}"},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR
                    )
                    
            except FileNotFoundError:
                return Response(
                    {"error": f"Data file not found at {DATA_FILE_PATH}"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            # 3. Parse the strategy
            try:
                parser = UniversalStrategyParser(strategy_json)
                parsed_strategy = parser.parse()
                print(f"Strategy parsed: {len(parsed_strategy['strategy_groups'])} groups, {len(parsed_strategy['indicators'])} indicators")
            except Exception as e:
                return Response(
                    {"error": f"Error parsing strategy: {str(e)}"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # 4. Run the backtest
            try:
                backtester = HybridBacktester(initial_capital=100000)
                results = backtester.run_backtest(data, parsed_strategy)
                
                print(f"Backtest completed:")
                print(f"  - Final capital: ${results['metrics']['final_capital']:,.2f}")
                print(f"  - Total trades: {results['metrics']['total_trades']}")
                print(f"  - Equity curve points: {len(results['equity_curve'])}")
                
            except Exception as e:
                print(f"Backtesting error: {str(e)}")
                return Response(
                    {"error": f"Error during backtesting: {str(e)}"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            # 5. Generate the interactive plot HTML
            try:
                plot_html = HybridVisualizationEngine.create_single_interactive_plot(results, data)
                print("Plot generated successfully")
            except Exception as e:
                print(f"Plot generation error: {str(e)}")
                plot_html = f"<div>Plot generation failed: {str(e)}</div>"
            
            # 6. Construct and send the final response
            final_response = {
                'metrics': results.get('metrics', {}),
                'plot_html': plot_html,
                'suggestions': results.get('suggestions', []),
            }
            return Response(final_response, status=status.HTTP_200_OK)

        except json.JSONDecodeError:
            return Response({"error": "Invalid JSON format in strategy."}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return Response({"error": f"An unexpected error occurred: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        

class TuningAPIView(APIView):
    """API endpoints for parameter tuning functionality"""
    
    def post(self, request):
        """
        Start a new parameter tuning job
        
        Expected payload:
        {
            "strategy_spec": {...},  # Base strategy configuration
            "param_space": {...},    # Parameter ranges to optimize
            "search_type": "grid|random",
            "budget": 50,           # Max number of combinations
            "objective": "sharpe_ratio"  # Metric to optimize
        }
        """
        try:
            # Validate required fields
            required_fields = ['strategy_spec', 'param_space', 'search_type']
            for field in required_fields:
                if field not in request.data:
                    return Response(
                        {'error': f'Missing required field: {field}'}, 
                        status=status.HTTP_400_BAD_REQUEST
                    )
            
            # Validate search_type
            if request.data['search_type'] not in ['grid', 'random']:
                return Response(
                    {'error': 'search_type must be either "grid" or "random"'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Validate param_space format
            param_space = request.data['param_space']
            if not self._validate_param_space(param_space):
                return Response(
                    {'error': 'Invalid param_space format'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Generate unique job ID
            job_id = f"tune_{uuid.uuid4().hex[:8]}"
            
            # Create tuning job record
            job = TuningJob.objects.create(
                job_id=job_id,
                strategy_spec=request.data['strategy_spec'],
                param_space=param_space,
                search_type=request.data['search_type'],
                budget=request.data.get('budget', 10),
                objective=request.data.get('objective', 'sharpe_ratio')
            )
            
            # Start tuning process in background thread
            tuner = TuningEngine()
            thread = threading.Thread(target=tuner.run_tuner, args=(job_id,))
            thread.daemon = True
            thread.start()
            
            logger.info(f"Started tuning job {job_id}")
            
            return Response({
                'job_id': job_id,
                'status': 'pending',
                'message': 'Tuning job started successfully'
            }, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            logger.error(f"Error starting tuning job: {str(e)}")
            return Response(
                {'error': f'Failed to start tuning job: {str(e)}'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def get(self, request):
        """
        Handle GET requests for tuning status and results
        """
        job_id = request.query_params.get('job_id')

        if not job_id:
            return Response(
                {'error': 'job_id parameter is required'}, 
                status=status.HTTP_400_BAD_REQUEST
            )

        # Correctly use get_object_or_404 without catching the exception here
        job = get_object_or_404(TuningJob, job_id=job_id)

        try:
            # Determine action based on URL path
            if 'status' in request.path:
                return self._get_job_status(job)
            elif 'results' in request.path:
                return self._get_job_results(job, request)
            else:
                return self._get_job_status(job)

        except Exception as e:
            # This block will now only catch true server-side errors, not missing objects
            logger.error(f"Error handling GET request: {str(e)}")
            return Response(
                {'error': str(e)}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def _get_job_status(self, job):
        """Get the current status and progress of a tuning job"""
        return Response({
            'job_id': job.job_id,
            'status': job.status,
            'progress': job.progress,
            'total_runs': job.total_runs,
            'progress_percentage': job.get_progress_percentage(),
            'created_at': job.created_at,
            'completed_at': job.completed_at,
            'search_type': job.search_type,
            'budget': job.budget,
            'objective': job.objective,
            'error_message': job.error_message
        })
    
    def _get_job_results(self, job, request):
        """Get the results of a completed tuning job"""
        if job.status != 'completed':
            return Response({
                'job_id': job.job_id,
                'status': job.status,
                'message': 'Job not yet completed',
                'results': []
            })
        
        top_n = int(request.query_params.get('top_n', 10))
        tuner = TuningEngine()
        
        try:
            results = tuner.get_best_results(job.job_id, top_n, job.objective)
            
            return Response({
                'job_id': job.job_id,
                'status': job.status,
                'total_runs': job.total_runs,
                'objective': job.objective,
                'top_results_count': len(results),
                'results': results
            })
            
        except Exception as e:
            return Response(
                {'error': f'Failed to get results: {str(e)}'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def _validate_param_space(self, param_space):
        """Validate the parameter space format"""
        if not isinstance(param_space, dict):
            return False
        
        for param_name, param_config in param_space.items():
            if not isinstance(param_config, dict):
                return False
            
            if 'type' not in param_config:
                return False
            
            param_type = param_config['type']
            
            if param_type in ['int', 'float']:
                if 'min' not in param_config or 'max' not in param_config:
                    return False
                if param_config['min'] >= param_config['max']:
                    return False
            elif param_type == 'choice':
                if 'choices' not in param_config or not param_config['choices']:
                    return False
            else:
                return False
        
        return True