from django.shortcuts import render

# Create your views here.
# summarizer/views.py

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import UrlSerializer
from .services import generate_summary_from_url

class SummaryAPIView(APIView):
    """
    API endpoint that accepts a YouTube URL and returns a summary.
    """
    def post(self, request, *args, **kwargs):
        serializer = UrlSerializer(data=request.data)
        
        if serializer.is_valid():
            url = serializer.validated_data['url']
            result = generate_summary_from_url(url)
            
            if "error" in result:
                return Response({"error": result["error"]}, status=status.HTTP_400_BAD_REQUEST)
            
            return Response(result, status=status.HTTP_200_OK)
            
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)