from django.urls import path
from . import views

urlpatterns = [
    # ...existing URLs...
    path("predict/", views.predict_view, name="predict"),
]