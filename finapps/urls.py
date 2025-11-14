from django.contrib import admin
from django.urls import path, include
from django.views.generic import TemplateView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('accounts/', include('apps.accounts.urls')),
    path('bank/', include('apps.banking.urls')),
    path('insights/', include('apps.insights.urls')),
    path('agents/', include('apps.agents.urls')),
    path('', TemplateView.as_view(template_name='index.html'), name='home'),
]
