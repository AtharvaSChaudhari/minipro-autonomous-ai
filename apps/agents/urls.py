from django.urls import path
from . import views

app_name = 'agents'

urlpatterns = [
    path('health/', views.health_agent, name='health'),
    path('loan/', views.loan_agent, name='loan'),
    path('invest/', views.invest_agent, name='invest'),
]
