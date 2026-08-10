import json
from pathlib import Path
from django.core.management.base import BaseCommand
from diagnosis.models import Disease


class Command(BaseCommand):
    help = "Seed disease data"

    def handle(self, *args, **kwargs):
        path = Path(__file__).resolve().parents[2] / "data" / "diseases.json"

        with open(path, encoding="utf-8") as file:
            diseases = json.load(file)

        for data in diseases:
            Disease.objects.update_or_create(
                model_class=data["model_class"],
                defaults={
                    "name_ar": data["name_ar"],
                    "name_en": data["name_en"],
                    "description": data["description"],
                    "treatment": data["treatment"],
                },
            )

        self.stdout.write(
            self.style.SUCCESS(f"تمت إضافة/تحديث {len(diseases)} مرضاً بنجاح.")
        )