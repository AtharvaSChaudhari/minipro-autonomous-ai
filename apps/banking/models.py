from django.db import models
from django.contrib.auth.models import User


class BankConnection(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='bank_connections')
    bank_name = models.CharField(max_length=100)
    account_number = models.CharField(max_length=50)
    ifsc_code = models.CharField(max_length=20)
    csv_url = models.URLField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.user.username} - {self.bank_name}"
