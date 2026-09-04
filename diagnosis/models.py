from django.db import models
from django.contrib.auth.models import User

# ==========================================================
# جدول الأمراض
# ==========================================================
class Disease(models.Model):
    name_ar = models.CharField(max_length=200)
    name_en = models.CharField(max_length=200)
    model_class = models.CharField(max_length=100, unique=True)
    description = models.TextField()
    treatment = models.JSONField()

    def __str__(self):
        return self.name_ar

# ==========================================================
# جدول التشخيص (التنبؤات) 
# ==========================================================
class Prediction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    image = models.ImageField(upload_to="predictions/")
    disease = models.ForeignKey(Disease, on_delete=models.CASCADE)
    confidence = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.disease.name_ar} ({self.confidence:.2%})"