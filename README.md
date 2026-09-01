# نظام ذكي لتشخيص أمراض أشجار الزيتون والحمضيات وتقديم توصيات المعالجة باستخدام التعلم العميق

## Deep Learning-Based Automated Diagnosis and Treatment Advisory Intelligent System for Olive and Citrus Diseases

نظام ويب ذكي يعتمد على التعلم العميق لتشخيص أمراض أشجار الزيتون والحمضيات من خلال تحليل صور الأوراق والثمار والأغصان والأجزاء المصابة من النبات.

### تم تطوير النظام باستخدام Python و Django وPyTorch، ويعتمد على نموذج EfficientNet-B0 المدرب مسبقاً على ImageNet والمعاد تدريبه لتصنيف 20 فئة من أمراض الزيتون والحمضيات.
---
## 📌 أهم وظائف النظام

* رفع صور النباتات المصابة وتحليلها.
* تشخيص المرض باستخدام EfficientNet-B0.
* عرض اسم المرض بالعربية والإنكليزية.
* عرض نسبة الثقة ووصف المرض.
* عرض توصيات المعالجة.
* حفظ نتائج التشخيص في سجل المستخدم.
* استعراض وحذف التشخيصات السابقة.
* تحميل تقارير PDF.
* إدارة المستخدمين والأمراض والتشخيصات من خلال Django Admin.

---

## 🧠 نموذج الذكاء الاصطناعي

EfficientNet-B0

تم استخدام Transfer Learning بالاعتماد على أوزان ImageNet، مع تعديل طبقة التصنيف النهائية لتناسب 20 فئة مرضية.

### الفئات المدعومة

أمراض الحمضيات:

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

أمراض الزيتون:

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

يتكون Dataset من:

* 2000 صورة
* 20 فئة
* 100 صورة لكل فئة

تم استخدام Stratified Split وفق النسب التالية:

| Dataset    | النسبة | عدد الصور |
| ---------- | -----: | --------: |
| Training   |    70% |      1400 |
| Validation |    15% |       300 |
| Testing    |    15% |       300 |

كما تم استخدام Data Augmentation لتحسين قدرة النموذج على التعميم.

---

## 🏆 نتائج EfficientNet-B0

| Metric              |     Result |
| ------------------- | ---------: |
| Test Accuracy       | 95.68% |
| Precision           | 95.97% |
| Recall              | 95.67% |
| F1 Score            | 95.68% |
| Validation Accuracy | 96.67% |


---

## 🔬 مقارنة النماذج

تمت مقارنة EfficientNet-B0 مع ResNet18:

| Model           | Validation Accuracy | Test Accuracy |   F1 Score |
| --------------- | ------------------: | ------------: | ---------: |
| EfficientNet-B0 |          96.67% |    95.68% | 95.68% |
| ResNet18        |          92.00% |    94.67% | 94.64% |

وبناءً على نتائج الاختبار، تم اعتماد EfficientNet-B0 كنموذج التصنيف الرئيسي للنظام.
---

## 🛠️ التقنيات المستخدمة

Backend

* Python
* Django
* PyTorch
* Torchvision
* SQLite / MySQL

Deep Learning

* EfficientNet-B0
* ResNet18
* Transfer Learning
* Data Augmentation
* Stratified Dataset Splitting

Image Processing

* Pillow
* OpenCV
* Albumentations

Frontend

* HTML5
* CSS3
* Bootstrap 5
* JavaScript
* Bootstrap Icons

PDF

* ReportLab
* Arabic Reshaper
* Python Bidi

---

## 📁 بنية المشروع


<img width="534" height="666" alt="image" src="https://github.com/user-attachments/assets/e571362d-6b3d-4380-b9da-cfdc77ddea3b" />


## ⚙️ تشغيل المشروع محليًا

### 1. استنساخ المستودع


git clone https://github.com/MohamadAlras/Olive_Citrus_Disease_Diagnosis_System.git
cd Olive_Citrus_Disease_Diagnosis_System

### 2. إنشاء البيئة الافتراضية

python -m venv venv

على Windows:

venv\Scripts\activate

### 3. تثبيت المتطلبات

pip install -r requirements.txt

### 4. تطبيق Migrations

python manage.py migrate

### 5. إضافة بيانات الأمراض

python manage.py seed_diseases

### 6. إنشاء مستخدم Administrator

python manage.py createsuperuser

### 7. تشغيل الخادم

python manage.py runserver

ثم افتح:

http://127.0.0.1:8000/

وللوصول إلى لوحة الإدارة:

http://127.0.0.1:8000/admin/
---

## 📦 الملفات الأساسية

* `models/efficientnet_b0_stratified_best.pth` — نموذج EfficientNet-B0 المستخدم في التشخيص.
* `seed_diseases.py` — تهيئة بيانات الأمراض وتوصيات المعالجة.
* `db.sqlite3` — قاعدة البيانات المستخدمة في بيئة التطوير المحلية.
* `media/predictions/` — تخزين صور التشخيص.
* `requirements.txt` — مكتبات ومتطلبات المشروع.

---

## ⚠️ ملاحظة

النظام مخصص للأغراض التجريبية، وتعتمد دقة التشخيص على جودة الصورة ومدى تشابهها مع البيانات المستخدمة في تدريب النموذج.

---

## 🎓 المشروع

المشروع عبارة عن رسالة ماجستير التأهيل والتخصص في علوم الحاسوب، ويجمع بين:

Deep Learning + Computer Vision + Web Development + Plant Disease Diagnosis

---

## 👨‍💻 Author

Mohamad Alras

GitHub:
https://github.com/MohamadAlras

---

## 📜 License

هذا المشروع مخصص للأغراض التجريبية المهنية.
