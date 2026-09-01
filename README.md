# نظام ذكي لتشخيص أمراض أشجار الزيتون والحمضيات وتقديم توصيات المعالجة باستخدام التعلم العميق

# Deep Learning-Based Automated Diagnosis and Treatment Advisory Intelligent System for Olive and Citrus Diseases

نظام ويب ذكي يعتمد على التعلم العميق (Deep Learning) لتشخيص أمراض أشجار الزيتون والحمضيات من خلال تحليل صور الأوراق والثمار والأغصان والأجزاء المصابة من النبات.

تم تطوير النظام باستخدام (Python وDjango وPyTorch)، ويستخدم نموذج (EfficientNet-B0) المدرب مسبقاً على ImageNet والمُعاد تدريبه لتصنيف 20 فئة من أمراض الزيتون والحمضيات.

## 📌 فكرة المشروع

يهدف المشروع إلى تطوير نظام يساعد المستخدم على:

- رفع صورة للنبات المصاب.
- تحليل الصورة باستخدام نموذج تعلم عميق Deep Learning.
- تحديد المرض المتوقع.
- عرض اسم المرض بالعربية والإنكليزية.
- عرض نسبة الثقة في التنبؤ.
- عرض وصف المرض.
- عرض توصيات المعالجة.
- حفظ نتائج التشخيص في سجل خاص بالمستخدم.
- استعراض سجل التشخيصات السابقة.
- حذف التشخيصات.
- تحميل تقرير PDF يحتوي على نتائج التشخيص.
- إدارة الأمراض وعمليات التشخيص من خلال Django Admin.

---

## 🧠 نموذج الذكاء الاصطناعي

تم استخدام نموذج:

**EfficientNet-B0**

مع الاستفادة من الأوزان المدربة مسبقاً على ImageNet، ثم استبدال طبقة التصنيف النهائية لتناسب عدد الفئات في المشروع.

عدد الفئات:

**20 فئة**

### الفئات المدعومة

#### 🍊 أمراض الحمضيات

1. Citrus Black Aphid
2. Citrus Black Spot
3. Citrus Brown Rot
4. Citrus Canker
5. Citrus Greening
6. Citrus Gummosis
7. Citrus Mealybugs
8. Citrus Mussel Scale
9. Citrus Penicillium Molds
10. Citrus Sooty Mold

#### 🌿 أمراض الزيتون

11. Olive Aculus Olearius
12. Olive Anthracnose
13. Olive Knot
14. Olive Leopard
15. Olive Bactrocera
16. Olive Prays Oleae
17. Olive Psyllid
18. Olive Saissetia Oleae
19. Olive Spilocaea Oleagina
20. Olive Verticillium Wilt

---

## 📊 بيانات التدريب

يتكون Dataset المستخدم في المشروع من:

- **2000 صورة**
- **20 فئة**
- **100 صورة لكل فئة**

تم استخدام **Stratified Split** لتقسيم البيانات إلى:

- بيانات التدريب (Training): 70%
- بيانات التحقق (Validation) 15%
- بيانات الاختبار (Testing) 15%

أي:

- Training: 1400 صورة
- Validation: 300 صورة
- Testing: 300 صورة

كما تم استخدام تقنيات زيادة البيانات Image Augmentation أثناء التدريب لتحسين قدرة النموذج على التعميم.

---

## 🏆 نتائج النموذج

حقق نموذج EfficientNet-B0 النتائج التالية على مجموعة الاختبار:

| Metric        |     Result |
| ------------- | ---------: |
| Test Accuracy | **95.68%** |
| Precision     | **95.97%** |
| Recall        | **95.67%** |
| F1 Score      | **95.68%** |

أفضل دقة تحقق:
**96.67% Validation Accuracy**

---

## 🔬 مقارنة النماذج

تمت مقارنة EfficientNet-B0 مع ResNet18.

| Model           | Validation Accuracy | Test Accuracy |   F1 Score |
| --------------- | ------------------: | ------------: | ---------: |
| EfficientNet-B0 |          **96.67%** |    **95.68%** | **95.68%** |
| ResNet18        |              92.00% |        94.67% |     94.64% |

وبناءً على نتائج الاختبار، تم اعتماد **EfficientNet-B0** كنموذج التصنيف الرئيسي للنظام.

---

## 🛠️ التقنيات المستخدمة

### Backend

- Python
- Django
- SQLite
- PyTorch
- Torchvision

### Deep Learning

- EfficientNet-B0
- ResNet18
- Transfer Learning
- Image Augmentation
- Stratified Dataset Splitting

### Image Processing

- Pillow
- OpenCV
- Albumentations

### Frontend

- HTML5
- CSS3
- Bootstrap 5
- Bootstrap Icons
- JavaScript
- Cairo Font

### PDF

- ReportLab
- Arabic Reshaper
- Python Bidi

---

## 👤 إدارة المستخدمين

يوفر النظام إدارة المستخدمين يتضمن:

- إنشاء حساب جديد.
- تسجيل الدخول باستخدام البريد الإلكتروني.
- تسجيل الخروج.
- تعديل الملف الشخصي.
- منع تكرار اسم المستخدم.
- منع تكرار البريد الإلكتروني.
- حماية صفحات التشخيص باستخدام Authentication.
- عرض سجل التشخيص الخاص بالمستخدم فقط.

---

## 🔐 Django Administration

يحتوي النظام على لوحة إدارة مخصصة لإدارة:

- الأمراض.
- بيانات الأمراض باللغة العربية والإنكليزية.
- أوصاف الأمراض.
- توصيات المعالجة.
- عمليات التشخيص.
- المستخدمين.
- نسب الثقة.
- تاريخ عمليات التشخيص.

كما تم تخصيص مظهر لوحة الإدارة ليتناسب مع الهوية البصرية للنظام.

---

## 📄 تقارير PDF

يمكن للمستخدم تحميل تقرير PDF يحتوي على:

- معلومات المستخدم.
- تاريخ إنشاء التقرير.
- صورة التشخيص.
- اسم المرض بالعربية.
- اسم المرض بالإنكليزية.
- نسبة الثقة.
- تاريخ التحليل.
- وصف المرض.
- توصيات المعالجة.

ويدعم التقرير عرض النصوص العربية باستخدام الخطوط المناسبة ومعالجة اتجاه النص العربي.

---

## 📁 بنية المشروع

OliveCitrusDiseaseSystem/
diagnosis/
data/
diseases.json
management/
commands/
seed_diseases.py
migrations/
admin.py
models.py
urls.py
views.py

OliveCitrusDiseaseSystem/
settings.py
urls.py
asgi.py
wsgi.py

models/
efficientnet_b0_stratified_best.pth
media/
predictions/

raw_olive_citrus_dataset/
citrus_black_aphid/
citrus_black_spot/
olive_verticillium_wilt/
static/
css/
images/
js/

templates/
admin/
pages/
parts/

db.sqlite3
manage.py
requirements.txt
README.md

## ⚙️ تشغيل المشروع محلياً

### 1. استنساخ المستودع

git clone https://github.com/MohamadAlras/Olive_Citrus_Disease_Diagnosis_System.git

ثم:

cd Olive_Citrus_Disease_Diagnosis_System

### 2. إنشاء Virtual Environment

python -m venv venv

تفعيل البيئة على Windows:

venv\Scripts\activate

### 3. تثبيت المتطلبات

pip install -r requirements.txt

### 4. تطبيق Migrations

python manage.py migrate

### 5. إنشاء بيانات الأمراض

python manage.py seed_diseases

### 6. إنشاء مستخدم Administrator

python manage.py createsuperuser

### 7. تشغيل الخادم

python manage.py runserver

بعد ذلك يمكن الوصول إلى النظام من:

http://127.0.0.1:8000/

ولوحة الإدارة من:

http://127.0.0.1:8000/admin/

---

## 📦 الملفات المهمة

### نموذج الذكاء الاصطناعي

models/efficientnet_b0_stratified_best.pth

وهو النموذج المستخدم فعلياً في عملية التشخيص.

### قاعدة البيانات المحلية:

db.sqlite3

### قاعدة البيانات عند النشر

MySQL

تحتوي على بيانات النظام الحالية، بما في ذلك المستخدمين والأمراض وسجل التشخيصات.

### مجموعة البيانات Dataset تحتوي صور التدريب والتحقق والاختبار

raw_olive_citrus_dataset/


---

## ⚠️ ملاحظات مهمة

هذا المشروع مخصص لأغراض **البحث والتجريب**.

تعتمد دقة التشخيص على جودة الصورة ومدى تشابهها مع البيانات المستخدمة في تدريب النموذج.

---

## 🎓 المشروع المهني

هذا المشروع جزء من **ماجستير في علوم الحاسوب**، ويجمع بين:

**Deep Learning + Computer Vision + Web Development + Plant Disease Diagnosis**

ويهدف إلى بناء نظام عملي لربط نموذج التصنيف العميق بتطبيق ويب يمكن للمستخدم التفاعل معه بسهولة.

---

## 👨‍💻 Author

**Mohamad Alras**

GitHub:

https://github.com/MohamadAlras

---

## 📜 License

هذا المشروع مخصص للاستخدام المهني.
