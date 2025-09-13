from django.urls import path
from .views import BacktestAPIView
from .views import TuningAPIView

urlpatterns = [
    path('backtest/', BacktestAPIView.as_view(), name='backtest'),
    path('tuner/run/', TuningAPIView.as_view(), name='tuner-run'),
    path('tuner/status/', TuningAPIView.as_view(), name='tuner-status'),
    path('tuner/results/', TuningAPIView.as_view(), name='tuner-results'),
]