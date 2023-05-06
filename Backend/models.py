from django.db import models

# Create your models here.
class Newuser(models.Model):
    Username=models.CharField(max_length=150)
    Email=models.EmailField(max_length=150)
    Password=models.CharField(max_length=150)
    ConfirmPassword=models.CharField(max_length=150)

class Contact(models.Model):
    name=models.CharField(max_length=150)
    email=models.EmailField(max_length=150)
    subject=models.TextField(max_length=1500)
    def __str__(self):
        return self.name