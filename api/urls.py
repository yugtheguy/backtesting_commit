from django.urls import path
from .views import BacktestAPIView

urlpatterns = [
    path('backtest/', BacktestAPIView.as_view(), name='backtest'),
]