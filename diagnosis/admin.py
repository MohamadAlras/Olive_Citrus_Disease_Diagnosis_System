from django.contrib import admin
from django.contrib.auth.models import User
from django.db.models import Avg
from django.utils import timezone

from .models import Disease, Prediction

# ==========================================================
# Disease Admin
# ==========================================================
@admin.register(Disease)
class DiseaseAdmin(admin.ModelAdmin):
    list_display = ("name_ar", "name_en", "model_class")
    search_fields = ("name_ar", "name_en", "model_class")

# ==========================================================
# Prediction Admin
# ==========================================================
@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ("disease", "confidence", "created_at", "user")
    list_filter = ("created_at",)
    search_fields = ("disease__name_ar", "disease__name_en")

    # ======================================================
    # إحصائيات لوحة الإدارة
    # ======================================================

    change_list_template = "admin/diagnosis/prediction/change_list.html"

    def changelist_view(self, request, extra_context=None):
        today = timezone.localdate()

        stats = {
            "total_users": User.objects.count(),
            "total_predictions": Prediction.objects.count(),
            "total_diseases": Disease.objects.count(),
            "average_confidence": (
                Prediction.objects.aggregate(
                    avg=Avg("confidence")
                )["avg"] or 0
            ),
            "today_predictions": Prediction.objects.filter(
                created_at__date=today
            ).count(),
        }
        extra_context = extra_context or {}
        extra_context["stats"] = stats

        return super().changelist_view(
            request,
            extra_context=extra_context,
        )