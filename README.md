# 🚗 Road Vehicle Image Processing Project (`imageprocessing-prj`)

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset: Kaggle](https://img.shields.io/badge/Kaggle-Road%20Vehicle%20Dataset-20BEFF.svg)](https://www.kaggle.com/datasets/ashfakyeafi/road-vehicle-images-dataset)
[![Format: YOLO](https://img.shields.io/badge/Format-YOLOv8%20%2F%20YOLOv11-orange.svg)](https://github.com/ultralytics/ultralytics)

โปรเจกต์ประมวลผลภาพขั้นสูงและการเตรียมชุดข้อมูลยานพาหนะบนท้องถนน 21 คลาส สำหรับงาน **Object Detection & Computer Vision** พร้อมระบบอัตโนมัติ (End-to-End Pipeline) ตั้งแต่การดาวน์โหลด, การวิเคราะห์ข้อมูลเชิงลึก (EDA), การปรับขนาดภาพแบบ Letterbox พร้อมคำนวณพิกัด Bounding Box ใหม่ และการแบ่งชุดข้อมูลแบบ Multi-Label Stratification (Train 70% / Val 15% / Test 15%)

---

## 📑 สารบัญ (Table of Contents)
- [โครงสร้างโปรเจกต์ (Project Structure)](#-โครงสร้างโปรเจกต์-project-structure)
- [ข้อมูลชุดข้อมูล (Dataset Overview)](#-ข้อมูลชุดข้อมูล-dataset-overview)
- [การติดตั้งและการเริ่มต้นใช้งาน (Getting Started)](#-การติดตั้งและการเริ่มต้นใช้งาน-getting-started)
- [การรัน Pipeline ทั้งหมดในคลิกเดียว (Automated Pipeline Execution)](#-การรัน-pipeline-ทั้งหมดในคลิกเดียว-automated-pipeline-execution)
- [การรันทีละขั้นตอน (Step-by-Step Execution)](#-การรันทีละขั้นตอน-step-by-step-execution)
- [สรุปรายงานโครงการ (Reports & Documentation)](#-สรุปรายงานโครงการ-reports--documentation)

---

## 📁 โครงสร้างโปรเจกต์ (Project Structure)

```
imageprocessing-prj/
├── data/                               # จัดเก็บข้อมูลรูปภาพ (ถูก ignore โดย Git)
│   ├── raw/                            # ข้อมูลดิบที่ดาวน์โหลดจาก Kaggle (3,004 ภาพ)
│   ├── processed/                      # ข้อมูลหลัง Preprocess (Letterbox 640x640)
│   └── split/                          # ข้อมูลแบ่ง Train (70%) / Val (15%) / Test (15%)
│       ├── train/ (images, labels)
│       ├── val/   (images, labels)
│       ├── test/  (images, labels)
│       └── dataset_split.yaml          # Master YOLO config สำหรับส่ง Train
├── reports/                            # โฟลเดอร์รวมรายงานและผลการวิเคราะห์
│   ├── eda/
│   │   └── eda_report.md               # รายงานผลการสำรวจและวิเคราะห์ข้อมูลเชิงลึก
│   ├── preprocessing/
│   │   └── preprocessing_report.md     # รายงานผลการทำ Image Preprocessing & Math
│   ├── splitting/
│   │   └── data_splitting_report.md    # รายงานผลการแบ่งชุดข้อมูล Train/Val/Test
│   └── README.md                       # รายงานสรุปผลภาพรวมทั้งโครงการ
├── src/                                # Source Code ทั้งหมดของโปรเจกต์
│   ├── data/
│   │   ├── download_data.py            # สคริปต์ดาวน์โหลดข้อมูลจาก Kaggle Hub
│   │   ├── split_data.py               # อัลกอริทึม Multi-Label Stratified Splitting
│   │   └── run_split.py                # CLI Runner สำหรับ Data Splitting
│   ├── eda/
│   │   ├── eda_analyzer.py             # ตัววิเคราะห์สถิติภาพและ Bounding Box เชิงลึก
│   │   └── run_eda.py                  # CLI Runner สำหรับ EDA
│   └── preprocessing/
│       ├── preprocessor.py             # ตัวแปลงภาพ Letterbox และคำนวณพิกัด BBox
│       └── run_preprocessing.py        # CLI Runner สำหรับ Preprocessing
├── .env.example                        # เทมเพลตเก็บ Kaggle API Key ปลอดภัย
├── .gitignore                          # กำหนดไฟล์/โฟลเดอร์ที่ไม่ให้ Push ขึ้น Git
├── download_data.py                    # Entrypoint สคริปต์ดาวน์โหลดข้อมูล
├── requirements.txt                    # รายการ Library Dependencies
├── run_pipeline.bat                    # สคริปต์รัน Pipeline ทั้งหมดบน Windows ในคำสั่งเดียว
├── run_pipeline.py                     # สคริปต์ Python รัน Master Pipeline
└── README.md                           # เอกสารกำกับโปรเจกต์
```

---

## 📌 ข้อมูลชุดข้อมูล (Dataset Overview)

- **ที่มา:** Kaggle Hub (`ashfakyeafi/road-vehicle-images-dataset`)
- **รูปแบบ:** YOLO Format (`[class, x_center, y_center, width, height]`)
- **จำนวนรูปภาพทั้งหมด:** **3,004** ภาพ
- **จำนวนวัตถุทั้งหมด:** **24,348** วัตถุ (เฉลี่ย 8.11 วัตถุ/ภาพ)
- **จำนวนคลาส (21 คลาส):**
  `ambulance`, `army vehicle`, `auto rickshaw`, `bicycle`, `bus`, `car`, `garbagevan`, `human hauler`, `minibus`, `minivan`, `motorbike`, `pickup`, `policecar`, `rickshaw`, `scooter`, `suv`, `taxi`, `three wheelers -CNG-`, `truck`, `van`, `wheelbarrow`

---

## 🚀 การติดตั้งและการเริ่มต้นใช้งาน (Getting Started)

### 1. Clone Repository
```bash
git clone https://github.com/PichyyyNews/imageprocessing-prj.git
cd imageprocessing-prj
```

### 2. ติดตั้ง Dependencies
```bash
pip install -r requirements.txt
```

### 3. ตั้งค่า Kaggle API Token (Optional)
```bash
cp .env.example .env
```
ใส่ Token ลงใน `.env`:
```env
KAGGLE_API_TOKEN=your_kaggle_api_token_here
```

---

## ⚡ การรัน Pipeline ทั้งหมดในคลิกเดียว (Automated Pipeline Execution)

### ตัวเลือกที่ 1: ผ่าน Windows Batch File (`.bat`)
ดับเบิลคลิกที่ไฟล์ **`run_pipeline.bat`** หรือรันผ่าน Command Prompt / PowerShell:
```cmd
run_pipeline.bat
```

### ตัวเลือกที่ 2: ผ่าน Python Master Runner
```bash
python run_pipeline.py
```

> **ขั้นตอนที่ Pipeline ดำเนินการให้อัตโนมัติ:**
> 1. ตรวจสอบสภาพแวดล้อมและ Library Dependencies
> 2. ดาวน์โหลดชุดข้อมูลจาก Kaggle Hub มายัง `data/raw/`
> 3. ทำการวิเคราะห์ EDA และสร้างรายงาน `reports/eda/eda_report.md`
> 4. ทำการ Preprocess ภาพเป็น $640 \times 640$ Letterbox พร้อมปรับพิกัด Bounding Box สู่ `data/processed/`
> 5. ทำการแบ่งข้อมูล Multi-Label Stratification เป็น Train (70%), Val (15%), Test (15%) สู่ `data/split/`

---

## 🛠️ การรันทีละขั้นตอน (Step-by-Step Execution)

### Step 1: ดาวน์โหลดข้อมูล
```bash
python download_data.py
```

### Step 2: รัน Exploratory Data Analysis (EDA)
```bash
python src/eda/run_eda.py --data-root data/raw --output-report reports/eda/eda_report.md
```

### Step 3: รัน Image Preprocessing Pipeline
```bash
python src/preprocessing/run_preprocessing.py --src-root data/raw --dst-root data/processed --width 640 --height 640
```

### Step 4: รัน Multi-Label Stratified Data Splitting
```bash
python src/data/run_split.py --src-root data/processed --dst-root data/split --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15 --seed 42
```

---

## 📑 สรุปรายงานโครงการ (Reports & Documentation)

| ขั้นตอน | เอกสารรายงาน | รายละเอียดที่สำคัญ |
| :--- | :--- | :--- |
| **ภาพรวมทั้งหมด** | [`reports/README.md`](reports/README.md) | สรุปภาพรวมสถาปัตยกรรมและผลลัพธ์ทั้ง 4 Phase |
| **Phase 1: EDA** | [`reports/eda/eda_report.md`](reports/eda/eda_report.md) | สถิติขนาดภาพ, การกระจายตัวของคลาส, Imbalance Ratio $1,824.67:1$, รูปร่าง BBox |
| **Phase 2: Preprocessing** | [`reports/preprocessing/preprocessing_report.md`](reports/preprocessing/preprocessing_report.md) | สูตรคณิตศาสตร์การแปลงพิกัด Letterbox, การ Clipping ขอบเขต, และการกรอง BBox |
| **Phase 3: Splitting** | [`reports/splitting/data_splitting_report.md`](reports/splitting/data_splitting_report.md) | การแก้ปัญหา 0-sample ใน Validation เดิม, การสร้าง Test Set, ตาราง Class Matrix |

---

## 👨‍💻 ผู้จัดทำ (Author)
- **GitHub:** [@PichyyyNews](https://github.com/PichyyyNews)
- **Email:** `pichayut030347@gmail.com`
