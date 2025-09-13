

# Create your models here.
# api/models.py - Add these models to your existing models.py file

from django.db import models
import json

class TuningJob(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    job_id = models.CharField(max_length=100, unique=True, primary_key=True)
    strategy_spec = models.JSONField()  # Store the strategy configuration
    param_space = models.JSONField()    # Store the parameter space definition
    search_type = models.CharField(max_length=20, choices=[('grid', 'Grid'), ('random', 'Random')])
    budget = models.IntegerField(default=10)  # Maximum number of parameter combinations to test
    objective = models.CharField(max_length=50, default='sharpe_ratio')  # Metric to optimize
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    progress = models.IntegerField(default=0)  # Number of completed runs
    total_runs = models.IntegerField(default=0)  # Total number of planned runs
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    error_message = models.TextField(blank=True, null=True)
    
    def __str__(self):
        return f"TuningJob {self.job_id} - {self.status}"
    
    def get_progress_percentage(self):
        if self.total_runs == 0:
            return 0
        return int((self.progress / self.total_runs) * 100)


class TuningRun(models.Model):
    job = models.ForeignKey(TuningJob, on_delete=models.CASCADE, related_name='runs')
    run_id = models.AutoField(primary_key=True)
    params = models.JSONField()  # The specific parameters used in this run
    metrics_json = models.JSONField()  # All backtest metrics
    equity_curve_data = models.JSONField()  # Equity curve as list of data points
    trades_data = models.JSONField(null=True, blank=True)  # Trade history
    created_at = models.DateTimeField(auto_now_add=True)
    execution_time = models.FloatField(null=True, blank=True)  # Time taken in seconds
    
    def __str__(self):
        return f"Run {self.run_id} for Job {self.job.job_id}"
    
    def get_objective_value(self, objective='sharpe_ratio'):
        """Get the value of the objective metric for this run"""
        return self.metrics_json.get(objective, 0)
    
    class Meta:
        ordering = ['-created_at']