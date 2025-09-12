# summarizer/serializers.py

from rest_framework import serializers

class UrlSerializer(serializers.Serializer):
    """
    Serializer to validate the incoming YouTube URL.
    """
    url = serializers.URLField(required=True)