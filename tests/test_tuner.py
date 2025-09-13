# tests/test_tuner.py - Comprehensive test cases for parameter tuner

import json
import time
import threading
from django.test import TestCase, TransactionTestCase
from django.urls import reverse
from rest_framework.test import APIClient
from rest_framework import status
from unittest.mock import patch, MagicMock
from api.models import TuningJob, TuningRun
from api.tuner import TuningEngine


class TuningEngineTest(TestCase):
    """Test cases for the TuningEngine class"""
    
    def setUp(self):
        self.engine = TuningEngine()
        
    def test_generate_grid_combinations_int_params(self):
        """Test grid generation with integer parameters"""
        param_space = {
            'param1': {'type': 'int', 'min': 1, 'max': 3},
            'param2': {'type': 'int', 'min': 10, 'max': 12}
        }
        
        combinations = self.engine.generate_parameter_combinations(param_space, 'grid', 100)
        
        expected = [
            {'param1': 1, 'param2': 10},
            {'param1': 1, 'param2': 11},
            {'param1': 1, 'param2': 12},
            {'param1': 2, 'param2': 10},
            {'param1': 2, 'param2': 11},
            {'param1': 2, 'param2': 12},
            {'param1': 3, 'param2': 10},
            {'param1': 3, 'param2': 11},
            {'param1': 3, 'param2': 12},
        ]
        
        self.assertEqual(len(combinations), 9)
        self.assertEqual(combinations, expected)
    
    def test_generate_grid_combinations_with_budget_limit(self):
        """Test that grid search respects budget limitations"""
        param_space = {
            'param1': {'type': 'int', 'min': 1, 'max': 10},
            'param2': {'type': 'int', 'min': 1, 'max': 10}
        }
        
        combinations = self.engine.generate_parameter_combinations(param_space, 'grid', 5)
        
        self.assertEqual(len(combinations), 5)
    
    def test_generate_random_combinations(self):
        """Test random parameter generation"""
        param_space = {
            'param1': {'type': 'int', 'min': 1, 'max': 10},
            'param2': {'type': 'float', 'min': 0.1, 'max': 1.0},
            'param3': {'type': 'choice', 'choices': ['a', 'b', 'c']}
        }
        
        combinations = self.engine.generate_parameter_combinations(param_space, 'random', 5)
        
        self.assertEqual(len(combinations), 5)
        
        for combo in combinations:
            self.assertIn('param1', combo)
            self.assertIn('param2', combo)
            self.assertIn('param3', combo)
            
            self.assertIsInstance(combo['param1'], int)
            self.assertIsInstance(combo['param2'], float)
            self.assertIn(combo['param3'], ['a', 'b', 'c'])
            
            self.assertTrue(1 <= combo['param1'] <= 10)
            self.assertTrue(0.1 <= combo['param2'] <= 1.0)
    
    def test_get_best_results(self):
        """Test retrieving best results from completed job"""
        # Create a test job
        job = TuningJob.objects.create(
            job_id='test_job',
            strategy_spec={'strategy': 'test'},
            param_space={'param1': {'type': 'int', 'min': 1, 'max': 5}},
            search_type='random',
            status='completed',
            objective='sharpe_ratio'
        )
        
        # Create test runs with different Sharpe ratios
        test_runs = [
            {'params': {'param1': 1}, 'sharpe_ratio': 0.5},
            {'params': {'param1': 2}, 'sharpe_ratio': 1.2},  # Best
            {'params': {'param1': 3}, 'sharpe_ratio': 0.8},
            {'params': {'param1': 4}, 'sharpe_ratio': 1.0},
            {'params': {'param1': 5}, 'sharpe_ratio': 0.3}
        ]
        
        for run_data in test_runs:
            TuningRun.objects.create(
                job=job,
                params=run_data['params'],
                metrics_json={'sharpe_ratio': run_data['sharpe_ratio']},
                equity_curve_data=[1, 1.1, 1.2],
                execution_time=1.0
            )
        
        # Get best results
        results = self.engine.get_best_results('test_job', top_n=3)
        
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]['params']['param1'], 2)  # Best Sharpe ratio
        self.assertEqual(results[0]['objective_value'], 1.2)


class TuningAPITest(TestCase):
    """Test cases for the Tuning API endpoints"""
    
    def setUp(self):
        self.client = APIClient()
        self.valid_payload = {
            'strategy_spec': {
                'strategy_type': 'ma_crossover',
                'symbol': 'AAPL'
            },
            'param_space': {
                'fast_ma': {'type': 'int', 'min': 5, 'max': 20},
                'slow_ma': {'type': 'int', 'min': 20, 'max': 50}
            },
            'search_type': 'random',
            'budget': 5,
            'objective': 'sharpe_ratio'
        }
    
    def test_create_tuning_job_success(self):
        """Test successful creation of a tuning job"""
        response = self.client.post('/api/tuner/run/', self.valid_payload, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertIn('job_id', response.data)
        self.assertEqual(response.data['status'], 'pending')
        
        # Verify job was created in database
        job_id = response.data['job_id']
        job = TuningJob.objects.get(job_id=job_id)
        self.assertEqual(job.search_type, 'random')
        self.assertEqual(job.budget, 5)
    
    def test_create_tuning_job_missing_fields(self):
        """Test error handling for missing required fields"""
        invalid_payload = {'strategy_spec': {}}
        
        response = self.client.post('/api/tuner/run/', invalid_payload, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('error', response.data)
    
    def test_create_tuning_job_invalid_search_type(self):
        """Test error handling for invalid search type"""
        invalid_payload = self.valid_payload.copy()
        invalid_payload['search_type'] = 'invalid'
        
        response = self.client.post('/api/tuner/run/', invalid_payload, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('search_type must be either', response.data['error'])
    
    def test_get_job_status(self):
        """Test retrieving job status"""
        # Create a test job
        job = TuningJob.objects.create(
            job_id='test_status_job',
            strategy_spec={'strategy': 'test'},
            param_space={'param1': {'type': 'int', 'min': 1, 'max': 5}},
            search_type='grid',
            status='running',
            progress=3,
            total_runs=10
        )
        
        response = self.client.get('/api/tuner/status/', {'job_id': 'test_status_job'})
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['job_id'], 'test_status_job')
        self.assertEqual(response.data['status'], 'running')
        self.assertEqual(response.data['progress'], 3)
        self.assertEqual(response.data['total_runs'], 10)
        self.assertEqual(response.data['progress_percentage'], 30)
    
    def test_get_job_results_completed(self):
        """Test retrieving results from completed job"""
        # Create completed job with results
        job = TuningJob.objects.create(
            job_id='test_results_job',
            strategy_spec={'strategy': 'test'},
            param_space={'param1': {'type': 'int', 'min': 1, 'max': 3}},
            search_type='grid',
            status='completed',
            total_runs=3,
            objective='sharpe_ratio'
        )
        
        # Add some results
        for i in range(3):
            TuningRun.objects.create(
                job=job,
                params={'param1': i + 1},
                metrics_json={'sharpe_ratio': (i + 1) * 0.5},
                equity_curve_data=[1, 1.1],
                execution_time=1.0
            )
        
        response = self.client.get('/api/tuner/results/', {
            'job_id': 'test_results_job',
            'top_n': 2
        })
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['status'], 'completed')
        self.assertEqual(len(response.data['results']), 2)
        # Results should be sorted by Sharpe ratio (descending)
        self.assertEqual(response.data['results'][0]['params']['param1'], 3)
    
    def test_get_job_results_not_completed(self):
        """Test retrieving results from incomplete job"""
        job = TuningJob.objects.create(
            job_id='test_incomplete_job',
            strategy_spec={'strategy': 'test'},
            param_space={'param1': {'type': 'int', 'min': 1, 'max': 3}},
            search_type='grid',
            status='running'
        )
        
        response = self.client.get('/api/tuner/results/', {'job_id': 'test_incomplete_job'})
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['status'], 'running')
        self.assertEqual(response.data['results'], [])
    
    def test_invalid_param_space_format(self):
        """Test validation of parameter space format"""
        invalid_payloads = [
            # Missing 'type' field
            {
                **self.valid_payload,
                'param_space': {'param1': {'min': 1, 'max': 10}}
            },
            # Invalid type
            {
                **self.valid_payload,
                'param_space': {'param1': {'type': 'invalid', 'min': 1, 'max': 10}}
            },
            # Missing min/max for int type
            {
                **self.valid_payload,
                'param_space': {'param1': {'type': 'int', 'min': 1}}
            },
            # Missing choices for choice type
            {
                **self.valid_payload,
                'param_space': {'param1': {'type': 'choice'}}
            }
        ]
        
        for invalid_payload in invalid_payloads:
            response = self.client.post('/api/tuner/run/', invalid_payload, format='json')
            self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
            self.assertIn('Invalid param_space format', response.data['error'])


class TuningIntegrationTest(TransactionTestCase):
    """Integration tests that test the full tuning workflow"""
    
    def setUp(self):
        self.client = APIClient()
    
    @patch('api.engine.HybridBacktester')
    def test_full_tuning_workflow_easy(self, mock_backtester_class):
        """EASY: Test a simple random search with small budget"""
        # Mock the backtester to return consistent results
        mock_backtester = MagicMock()
        mock_backtester_class.return_value = mock_backtester
        
        def mock_backtest_results(strategy_config):
            # Return different results based on parameters
            ma_period = strategy_config.get('ma_period', 10)
            sharpe = 0.5 + (ma_period * 0.1)  # Simple relationship
            
            return {
                'metrics': {
                    'sharpe_ratio': sharpe,
                    'total_return_pct': sharpe * 20,
                    'max_drawdown_pct': 10,
                    'total_trades': 50
                },
                'equity_curve': [1.0, 1.1, 1.2],
                'trades': []
            }
        
        mock_backtester.run_backtest.side_effect = mock_backtest_results
        
        # Create tuning job
        payload = {
            'strategy_spec': {
                'strategy_type': 'ma_crossover',
                'symbol': 'AAPL'
            },
            'param_space': {
                'ma_period': {'type': 'int', 'min': 10, 'max': 50}
            },
            'search_type': 'random',
            'budget': 5,
            'objective': 'sharpe_ratio'
        }
        
        # Start tuning job
        response = self.client.post('/api/tuner/run/', payload, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        job_id = response.data['job_id']
        
        # Wait for job to complete
        max_wait = 10  # seconds
        wait_time = 0
        while wait_time < max_wait:
            status_response = self.client.get('/api/tuner/status/', {'job_id': job_id})
            if status_response.data['status'] == 'completed':
                break
            time.sleep(0.5)
            wait_time += 0.5
        
        # Verify job completed
        self.assertEqual(status_response.data['status'], 'completed')
        self.assertEqual(status_response.data['total_runs'], 5)
        
        # Get results
        results_response = self.client.get('/api/tuner/results/', {'job_id': job_id})
        self.assertEqual(results_response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(results_response.data['results']), 5)
        
        # Verify results are sorted by Sharpe ratio
        results = results_response.data['results']
        for i in range(len(results) - 1):
            self.assertGreaterEqual(
                results[i]['objective_value'],
                results[i + 1]['objective_value']
            )
        
        # Verify TuningRun records were created
        job = TuningJob.objects.get(job_id=job_id)
        runs = TuningRun.objects.filter(job=job)
        self.assertEqual(runs.count(), 5)
    
    @patch('api.engine.HybridBacktester')
    def test_grid_search_with_budget_limit(self, mock_backtester_class):
        """INTERMEDIATE: Test grid search with budget limitation"""
        mock_backtester = MagicMock()
        mock_backtester_class.return_value = mock_backtester
        mock_backtester.run_backtest.return_value = {
            'metrics': {'sharpe_ratio': 1.0, 'total_trades': 30},
            'equity_curve': [1.0, 1.1],
            'trades': []
        }
        
        # Create param space that would generate more combinations than budget
        payload = {
            'strategy_spec': {'strategy': 'test'},
            'param_space': {
                'param1': {'type': 'int', 'min': 1, 'max': 10},  # 10 values
                'param2': {'type': 'int', 'min': 1, 'max': 10}   # 10 values = 100 total combinations
            },
            'search_type': 'grid',
            'budget': 20,  # Limit to 20
            'objective': 'sharpe_ratio'
        }
        
        response = self.client.post('/api/tuner/run/', payload, format='json')
        job_id = response.data['job_id']
        
        # Wait for completion
        max_wait = 10
        wait_time = 0
        while wait_time < max_wait:
            status_response = self.client.get('/api/tuner/status/', {'job_id': job_id})
            if status_response.data['status'] == 'completed':
                break
            time.sleep(0.5)
            wait_time += 0.5
        
        # Verify that only budget number of runs were executed
        job = TuningJob.objects.get(job_id=job_id)
        runs = TuningRun.objects.filter(job=job)
        self.assertEqual(runs.count(), 20)  # Should not exceed budget
    
    @patch('api.engine.HybridBacktester')
    def test_error_handling_invalid_parameters(self, mock_backtester_class):
        """ADVANCED: Test error handling with invalid parameters"""
        mock_backtester = MagicMock()
        mock_backtester_class.return_value = mock_backtester
        
        # Make backtester raise an error for certain parameter values
        def mock_backtest_with_errors(strategy_config):
            ma_period = strategy_config.get('ma_period', 10)
            if ma_period < 0:  # Invalid parameter
                raise ValueError("MA period cannot be negative")
            
            return {
                'metrics': {'sharpe_ratio': 1.0, 'total_trades': 30},
                'equity_curve': [1.0, 1.1],
                'trades': []
            }
        
        mock_backtester.run_backtest.side_effect = mock_backtest_with_errors
        
        # Create job that might generate invalid parameters
        payload = {
            'strategy_spec': {'strategy': 'test'},
            'param_space': {
                'ma_period': {'type': 'int', 'min': -5, 'max': 20}  # Include negative values
            },
            'search_type': 'grid',
            'budget': 10,
            'objective': 'sharpe_ratio'
        }
        
        response = self.client.post('/api/tuner/run/', payload, format='json')
        job_id = response.data['job_id']
        
        # Wait for completion
        max_wait = 10
        wait_time = 0
        while wait_time < max_wait:
            status_response = self.client.get('/api/tuner/status/', {'job_id': job_id})
            if status_response.data['status'] in ['completed', 'failed']:
                break
            time.sleep(0.5)
            wait_time += 0.5
        
        # Job should complete even with some failed runs
        self.assertIn(status_response.data['status'], ['completed', 'failed'])
        
        # Check that some runs have error entries
        job = TuningJob.objects.get(job_id=job_id)
        runs = TuningRun.objects.filter(job=job)
        
        # Should have both successful and failed runs
        successful_runs = runs.exclude(metrics_json__has_key='error')
        failed_runs = runs.filter(metrics_json__has_key='error')
        
        self.assertGreater(successful_runs.count(), 0)
        self.assertGreater(failed_runs.count(), 0)


class TuningModelTest(TestCase):
    """Test cases for TuningJob and TuningRun models"""
    
    def test_tuning_job_progress_percentage(self):
        """Test progress percentage calculation"""
        job = TuningJob.objects.create(
            job_id='test_progress',
            strategy_spec={},
            param_space={},
            search_type='grid',
            total_runs=10,
            progress=3
        )
        
        self.assertEqual(job.get_progress_percentage(), 30)
        
        # Test edge case: no total runs
        job.total_runs = 0
        self.assertEqual(job.get_progress_percentage(), 0)
    
    def test_tuning_run_objective_value(self):
        """Test getting objective value from TuningRun"""
        job = TuningJob.objects.create(
            job_id='test_obj',
            strategy_spec={},
            param_space={},
            search_type='grid'
        )
        
        run = TuningRun.objects.create(
            job=job,
            params={'test': 1},
            metrics_json={
                'sharpe_ratio': 1.5,
                'total_return_pct': 25.0,
                'max_drawdown_pct': 8.0
            },
            equity_curve_data=[1.0, 1.1, 1.2]
        )
        
        self.assertEqual(run.get_objective_value('sharpe_ratio'), 1.5)
        self.assertEqual(run.get_objective_value('total_return_pct'), 25.0)
        self.assertEqual(run.get_objective_value('nonexistent'), 0)


# Usage instructions for running tests:
"""
To run these tests, use the following Django management commands:

# Run all tuner tests
python manage.py test tests.test_tuner

# Run specific test classes
python manage.py test tests.test_tuner.TuningEngineTest
python manage.py test tests.test_tuner.TuningAPITest
python manage.py test tests.test_tuner.TuningIntegrationTest

# Run specific test methods
python manage.py test tests.test_tuner.TuningEngineTest.test_generate_grid_combinations_int_params

# Run with verbose output
python manage.py test tests.test_tuner -v 2

# Run with coverage (if django-coverage is installed)
coverage run --source='.' manage.py test tests.test_tuner
coverage report
"""