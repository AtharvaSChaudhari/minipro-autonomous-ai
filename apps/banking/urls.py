from django.urls import path
from . import views

app_name = 'bank'

urlpatterns = [
    path('connect/', views.connect, name='connect'),
]
