from pathlib import Path
import os
import re
import torch
from PIL import Image

from django.conf import settings
from django.contrib import messages
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.shortcuts import get_object_or_404, redirect, render
from django.http import HttpResponse
from django.utils import timezone

from torchvision import models, transforms

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import cm
from reportlab.platypus import Image as ReportLabImage

import arabic_reshaper
from bidi.algorithm import get_display

from .models import Disease, Prediction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# أسماء الفئات بنفس ترتيب التدريب
# ==========================================================

CLASS_NAMES = [
    "citrus_black_aphid",
    "citrus_black_spot",
    "citrus_brown_rot",
    "citrus_canker",
    "citrus_greening",
    "citrus_gummosis",
    "citrus_mealybugs",
    "citrus_mussel_scale",
    "citrus_penicillium_molds",
    "citrus_sooty_mold",
    "olive_aculus_olearius",
    "olive_anthracnose",
    "olive_knot",
    "olive_leopard",
    "olive_pactrocerae",
    "olive_prays_oleae",
    "olive_psylle",
    "olive_saissetia_oleae",
    "olive_spilocaea_oleagina",
    "olive_verticillium_wilt",
]

# ==========================================================
# EfficientNet-B0 تحميل نموذج
# ==========================================================

model = models.efficientnet_b0(weights=None)
model.classifier[1] = torch.nn.Linear(
    model.classifier[1].in_features, 20
)

model.load_state_dict(
    torch.load(
        Path(settings.BASE_DIR)
        / "models"
        / "efficientnet_b0_stratified_best.pth",
        map_location=device,
    )
)

model.to(device).eval()

# ==========================================================
# معالجة الصور قبل التنبؤ
# ==========================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])

# ==========================================================
# PDF تجهيز النص العربي للعرض داخل
# ==========================================================

def arabic_text(text):
    if not text:
        return ""

    reshaped_text = arabic_reshaper.reshape(str(text))
    return get_display(reshaped_text)

# ==========================================================
# الصفحة الرئيسية
# ==========================================================

def home(request):
    return render(request, "pages/home.html")

# ==========================================================
# تسجيل الدخول باستخدام البريد الإلكتروني
# ==========================================================

def login_view(request):
    if request.method == "POST":
        email = request.POST.get("email", "").strip()
        password = request.POST.get("password", "")

        user = User.objects.filter(email__iexact=email).first()

        if user and user.check_password(password):
            login(request, user)
            return redirect("diagnosis:home")

        messages.error(
            request,
            "البريد الإلكتروني أو كلمة المرور غير صحيحة.",
        )

    return render(request, "pages/login.html")

# ==========================================================
# إنشاء حساب
# ==========================================================

def register(request):
    if request.method == "POST":
        data = request.POST
        first_name = data.get("first_name", "").strip()
        last_name = data.get("last_name", "").strip()
        username = data.get("username", "").strip()
        email = data.get("email", "").strip()
        password1 = data.get("password1", "")
        password2 = data.get("password2", "")

        # التحقق من تطابق كلمتي المرور
        if password1 != password2:
            messages.error(request, "كلمتا المرور غير متطابقتين.")
            return render(
                request,
                "pages/register.html",
                {
                    "first_name": first_name,
                    "last_name": last_name,
                    "username": username,
                    "email": email,
                },
            )

        # التحقق من طول كلمة المرور
        if len(password1) < 6:
            messages.error(request, "يجب أن تتكون كلمة المرور من 6 محارف على الأقل.")
            return render(
                request,
                "pages/register.html",
                {
                    "first_name": first_name,
                    "last_name": last_name,
                    "username": username,
                    "email": email,
                },
            )

        # التحقق من احتواء كلمة المرور على حروف وأرقام
        if not re.search(r"[A-Za-z]", password1) or not re.search(r"\d", password1):
            messages.error(request, "يجب أن تحتوي كلمة المرور على حروف وأرقام.")
            return render(
                request,
                "pages/register.html",
                {
                    "first_name": first_name,
                    "last_name": last_name,
                    "username": username,
                    "email": email,
                },
            )

        # التحقق من عدم تكرار اسم المستخدم
        if User.objects.filter(username=username).exists():
            messages.error(request, "اسم المستخدم مستخدم بالفعل.")
            return render(
                request,
                "pages/register.html",
                {
                    "first_name": first_name,
                    "last_name": last_name,
                    "username": username,
                    "email": email,
                },
            )

        # التحقق من عدم تكرار البريد الإلكتروني
        if User.objects.filter(email__iexact=email).exists():
            messages.error(request, "البريد الإلكتروني مستخدم بالفعل.")
            return render(
                request,
                "pages/register.html",
                {
                    "first_name": first_name,
                    "last_name": last_name,
                    "username": username,
                    "email": email,
                },
            )

        # إنشاء المستخدم
        User.objects.create_user(
            username=username,
            email=email,
            password=password1,
            first_name=first_name,
            last_name=last_name,
        )

        messages.success(request, "تم إنشاء الحساب بنجاح.")
        return redirect("diagnosis:login")

    return render(request, "pages/register.html")

# ==========================================================
# تسجيل الخروج
# ==========================================================

def logout_view(request):
    logout(request)
    return redirect("diagnosis:home")

# ==========================================================
# رفع الصورة وتحليلها
# ==========================================================

@login_required
def upload_image(request):
    if request.method == "POST":
        if "image" not in request.FILES:
            messages.error(request, "يرجى اختيار صورة.")
            return redirect("diagnosis:upload")

        image = request.FILES["image"]

        # 5MB التحقق من حجم الصورة — الحد الأقصى
        if image.size > 5 * 1024 * 1024:
            messages.error(
                request,
                "حجم الصورة كبير جداً. الحد الأقصى المسموح به هو 5 ميجابايت.")
            return redirect("diagnosis:upload")

        try:
            img = Image.open(image).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(x)
                probabilities = torch.softmax(output, dim=1)
                confidence, prediction = torch.max(
                    probabilities, dim=1
                )

            model_class = CLASS_NAMES[prediction.item()]

            disease = get_object_or_404(
                Disease,
                model_class=model_class,
            )

            image.seek(0)

            prediction = Prediction.objects.create(
                user=request.user,
                image=image,
                disease=disease,
                confidence=round(
                    confidence.item() * 100,
                    2,
                ),
            )

            return redirect("diagnosis:result", pk=prediction.pk,)

        except Exception as e:
            messages.error(request, f"حدث خطأ أثناء التحليل: {e}",)

    return render(request, "pages/upload.html")

# ==========================================================
# عرض نتيجة التشخيص
# ==========================================================

@login_required
def result(request, pk):
    prediction = get_object_or_404(
        Prediction,
        pk=pk,
        user=request.user,
    )

    return render(
        request,
        "pages/result.html",
        {"prediction": prediction},
    )
# ==========================================================
# سجل التشخيص
# ==========================================================

@login_required
def history(request):
    predictions = Prediction.objects.filter(
        user=request.user
    ).order_by("-created_at")

    return render(
        request,
        "pages/history.html",
        {"predictions": predictions},
    )

# ==========================================================
# PDF تحميل تقرير
# ==========================================================

@login_required
def download_pdf(request):

    predictions = Prediction.objects.filter(user=request.user).select_related("disease").order_by("-created_at")

    response = HttpResponse(content_type="application/pdf")
    response["Content-Disposition"] = 'attachment; filename="diagnosis_report.pdf"'

    # ------------------------------------------------------
    # إنشاء ملف PDF
    # ------------------------------------------------------

    pdf = canvas.Canvas(response, pagesize=A4)
    width, height = A4

    # ------------------------------------------------------
    # تحميل خط يدعم اللغة العربية
    # ------------------------------------------------------

    font_path = r"C:\Windows\Fonts\arial.ttf"
    pdfmetrics.registerFont(TTFont("ArabicFont", font_path))

    # ------------------------------------------------------
    # دالة بدء صفحة جديدة
    # ------------------------------------------------------

    def new_page():
        nonlocal y
        pdf.showPage()
        pdf.setFont("ArabicFont", 13)
        y = height - 2 * cm

    # ------------------------------------------------------
    # دالة التأكد من وجود مساحة
    # ------------------------------------------------------

    def check_space(required_height):
        if y < required_height:
            new_page()

    # ------------------------------------------------------
    # عنوان التقرير
    # ------------------------------------------------------

    y = height - 2 * cm

    pdf.setFont("ArabicFont", 18)
    pdf.drawCentredString(width / 2, y, arabic_text("تقرير تشخيص أمراض النباتات"))

    y -= 1.3 * cm

    # ------------------------------------------------------
    # معلومات المستخدم
    # ------------------------------------------------------

    pdf.setFont("ArabicFont", 14)

    user_name = f"{request.user.first_name} {request.user.last_name}".strip()

    if not user_name:
        user_name = request.user.username

    pdf.drawRightString(width - 2 * cm, y, arabic_text(f"المستخدم: {user_name}"))

    y -= 0.7 * cm

    pdf.drawRightString(width - 2 * cm, y, arabic_text(f"البريد الإلكتروني: {request.user.email}"))

    y -= 0.7 * cm

    report_date = timezone.localtime().strftime("%Y-%m-%d %H:%M")
    pdf.drawRightString(width - 2 * cm, y, arabic_text(f"تاريخ إنشاء التقرير: {report_date}"))

    y -= 1 * cm

    # ------------------------------------------------------
    # حالة عدم وجود تشخيصات
    # ------------------------------------------------------

    if not predictions:

        pdf.setFont("ArabicFont", 14)
        pdf.drawCentredString(width / 2, y, arabic_text("لا توجد سجلات تشخيص متاحة."))
        pdf.save()

        return response

    # ======================================================
    # عرض التشخيصات
    # ======================================================

    for index, prediction in enumerate(predictions, start=1):

        # --------------------------------------------------
        # عنوان التشخيص
        # --------------------------------------------------

        check_space(3 * cm)

        pdf.setFont("ArabicFont", 16)
        pdf.drawRightString(width - 2 * cm, y, arabic_text(f"التشخيص رقم {index}"))

        y -= 0.9 * cm

        # --------------------------------------------------
        # صورة التشخيص
        # --------------------------------------------------

        image_path = prediction.image.path

        if os.path.exists(image_path):

            try:

                image_width = 5 * cm
                image_height = 5 * cm

                check_space(image_height + 1.5 * cm)

                # الصورة في الجهة اليمنى
                image_x = width - 2 * cm - image_width
                image_y = y - image_height

                pdf.drawImage(
                    image_path,
                    image_x,
                    image_y,
                    width=image_width,
                    height=image_height,
                    preserveAspectRatio=True,
                    anchor="sw"
                )

                y -= image_height + 0.7 * cm

            except Exception:
                pass

        # --------------------------------------------------
        # اسم المرض بالعربية
        # --------------------------------------------------

        check_space(1.2 * cm)

        pdf.setFont("ArabicFont", 13)
        pdf.drawRightString(width - 2 * cm, y, arabic_text(f"اسم المرض: {prediction.disease.name_ar}"))

        y -= 0.7 * cm

        # --------------------------------------------------
        # اسم المرض بالإنكليزية
        # --------------------------------------------------

        pdf.drawRightString(width - 2 * cm, y, f"Disease Name: {prediction.disease.name_en}")

        y -= 0.7 * cm

        # --------------------------------------------------
        # نسبة الثقة
        # --------------------------------------------------

        pdf.drawRightString(width - 2 * cm, y, arabic_text(f"نسبة الثقة: {prediction.confidence:.2f}%"))

        y -= 0.7 * cm

        # --------------------------------------------------
        # تاريخ التحليل
        # --------------------------------------------------

        created_at = timezone.localtime(prediction.created_at).strftime("%Y-%m-%d %H:%M")

        pdf.drawRightString(width - 2 * cm, y, arabic_text(f"تاريخ التحليل: {created_at}"))

        y -= 0.9 * cm

        # ==================================================
        # وصف المرض
        # ==================================================

        check_space(3 * cm)

        pdf.setFont("ArabicFont", 13)
        pdf.drawRightString(width - 2 * cm, y, arabic_text("وصف المرض:"))

        y -= 0.6 * cm

        description = str(prediction.disease.description)
        words = description.split()
        line = ""

        for word in words:

            test_line = f"{line} {word}" if line else word

            if pdf.stringWidth(arabic_text(test_line), "ArabicFont", 13) > width - 4 * cm:

                if line:

                    check_space(0.8 * cm)

                    pdf.drawRightString(width - 2 * cm, y, arabic_text(line))
                    y -= 0.55 * cm

                line = word

            else:

                line = test_line

        if line:

            check_space(0.8 * cm)

            pdf.drawRightString(width - 2 * cm, y, arabic_text(line))
            y -= 0.8 * cm

        # ==================================================
        # توصيات المعالجة
        # ==================================================

        check_space(3 * cm)

        pdf.setFont("ArabicFont", 13)
        pdf.drawRightString(width - 2 * cm, y, arabic_text("توصيات المعالجة:"))

        y -= 0.6 * cm

        for item in prediction.disease.treatment:

            text = f"• {item}"
            words = str(text).split()
            line = ""

            for word in words:

                test_line = f"{line} {word}" if line else word

                if pdf.stringWidth(arabic_text(test_line), "ArabicFont", 13) > width - 4 * cm:

                    if line:

                        check_space(0.8 * cm)

                        pdf.drawRightString(width - 2 * cm, y, arabic_text(line))
                        y -= 0.55 * cm

                    line = word

                else:

                    line = test_line

            if line:

                check_space(0.8 * cm)

                pdf.drawRightString(width - 2 * cm, y, arabic_text(line))
                y -= 0.55 * cm

            y -= 0.1 * cm

        # --------------------------------------------------
        # فاصل بين التشخيصات
        # --------------------------------------------------

        check_space(1.5 * cm)

        y -= 0.4 * cm
        pdf.line(2 * cm, y, width - 2 * cm, y)
        y -= 0.9 * cm

    # ======================================================
    # PDF إنهاء ملف
    # ======================================================

    pdf.save()

    return response

# ==========================================================
# الملف الشخصي
# ==========================================================

@login_required
def profile(request):
    user = request.user

    if request.method == "POST":
        username = request.POST["username"].strip()
        email = request.POST["email"].strip()

        if User.objects.filter(
            username=username
        ).exclude(pk=user.pk).exists():
            messages.error(request, "اسم المستخدم مستخدم بالفعل.")
            return redirect("diagnosis:profile")

        if User.objects.filter(
            email__iexact=email
        ).exclude(pk=user.pk).exists():
            messages.error(request, "البريد الإلكتروني مستخدم بالفعل.")
            return redirect("diagnosis:profile")

        user.username = username
        user.email = email
        user.first_name = request.POST["first_name"].strip()
        user.last_name = request.POST["last_name"].strip()
        user.save()

        messages.success(request, "تم حفظ تعديلات الملف الشخصي بنجاح.")
        return redirect("diagnosis:profile")

    return render(request, "pages/profile.html")

# ==========================================================
# حذف تشخيص
# ==========================================================

@login_required
def delete_prediction(request, pk):
    prediction = get_object_or_404(
        Prediction,
        pk=pk,
        user=request.user,
    )

    if request.method == "POST":
        prediction.delete()
        messages.success(request, "تم حذف التشخيص بنجاح.")

    return redirect("diagnosis:history")