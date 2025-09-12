# summarizer/urls.py

from django.urls import path
from .views import SummaryAPIView

urlpatterns = [
    path('youtube/', SummaryAPIView.as_view(), name='summarize-video'),
]