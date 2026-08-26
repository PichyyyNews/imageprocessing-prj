# Road Vehicle Image Processing Project (`imageprocessing-prj`)

โปรเจกต์ประมวลผลภาพและตรวจจับประเภทยานพาหนะบนท้องถนน (Road Vehicle Object Detection & Image Processing)

## 📁 โครงสร้างโปรเจกต์ (Project Structure)
```
imageprocessing-prj/
├── data/                      # ที่เก็บข้อมูลรูปภาพ (ถูก ignore โดย Git)
│   ├── raw/                   # ข้อมูลดิบที่ดาวน์โหลดจาก Kaggle
│   └── processed/             # ข้อมูลที่ผ่านการ Preprocess
├── reports/                   # รายงานผลการวิเคราะห์และทดลอง (.md)
│   ├── eda/                   # รายงาน Exploratory Data Analysis
│   ├── preprocessing/         # รายงาน Image Preprocessing
│   └── splitting/             # รายงาน Data Splitting
├── src/                       # Source code หลักของโปรเจกต์
│   ├── data/                  # จัดการดาวน์โหลดและเตรียมข้อมูล
│   ├── eda/                   # การสำรวจและวิเคราะห์ข้อมูลเชิงลึก
│   └── preprocessing/         # Pipeline ประมวลผลภาพและปรับพิกัด Annotation
├── .env.example               # ตัวอย่างไฟล์ Environment Variables
├── .gitignore                 # ระบุไฟล์/โฟลเดอร์ที่ไม่ต้องการ Push ขึ้น Git
├── download_data.py           # Entrypoint สคริปต์ดาวน์โหลดข้อมูล
├── requirements.txt           # Dependencies ของโปรเจกต์
└── README.md                  # เอกสารกำกับโปรเจกต์
```

---

## 📌 ข้อมูล Dataset
- **ที่มา:** Kaggle Hub (`ashfakyeafi/road-vehicle-images-dataset`)
- **รูปแบบ:** YOLO Format (`data_1.yaml`, `train`, `valid`)
- **จำนวนรูปภาพทั้งหมด:** 3,004 ภาพ (Train: 2,704 ภาพ, Validation: 300 ภาพ)
- **จำนวนวัตถุทั้งหมด:** 24,348 วัตถุ (เฉลี่ย 8.11 วัตถุ/ภาพ)
- **จำนวนคลาส:** 21 คลาสยานพาหนะทางถนน

---

## 🚀 วิธีการใช้งาน

### 1. ติดตั้ง Dependencies
```bash
pip install -r requirements.txt
```

### 2. ตั้งค่า Kaggle API Token (Optional)
```bash
cp .env.example .env
```

### 3. ดาวน์โหลด Dataset
```bash
python download_data.py
```

### 4. รัน Exploratory Data Analysis (EDA)
```bash
python src/eda/run_eda.py
```
> รายงานฉบับเต็มจะถูกบันทึกไว้ที่ [`reports/eda/eda_report.md`](reports/eda/eda_report.md)

### 5. รัน Image Preprocessing Pipeline
```bash
python src/preprocessing/run_preprocessing.py
```
> ข้อมูลที่ผ่านการปรับขนาด Letterbox $640 \times 640$ และแปลงพิกัด Bounding Box จะถูกบันทึกไว้ที่ `data/processed/` และรายงานผลฉบับเต็มจะถูกบันทึกไว้ที่ [`reports/preprocessing/preprocessing_report.md`](reports/preprocessing/preprocessing_report.md)
