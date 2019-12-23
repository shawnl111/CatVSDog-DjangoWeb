from django.db import models

# Create your models here.
class image(models.Model):

    photo = models.ImageField(null=True, blank=True)

    def __str__(self):
        return self.photo.name
