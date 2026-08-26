# Road Vehicle Image Processing Project (`imageprocessing-prj`)

โปรเจกต์ประมวลผลภาพและตรวจจับประเภทยานพาหนะบนท้องถนน (Road Vehicle Object Detection & Image Processing)

## 📌 ข้อมูล Dataset
- **ที่มา:** Kaggle Hub (`ashfakyeafi/road-vehicle-images-dataset`)
- **รูปแบบ:** YOLO Format (`data_1.yaml`, `train`, `valid`)
- **จำนวนรูปภาพทั้งหมด:** 3,004 ภาพ (Train: 2,704 ภาพ, Validation: 300 ภาพ)
- **จำนวนคลาส:** 21 คลาส
  - `ambulance`, `army vehicle`, `auto rickshaw`, `bicycle`, `bus`, `car`, `garbagevan`, `human hauler`, `minibus`, `minivan`, `motorbike`, `pickup`, `policecar`, `rickshaw`, `scooter`, `suv`, `taxi`, `three wheelers -CNG-`, `truck`, `van`, `wheelbarrow`

---

## 🚀 การติดตั้งและดาวน์โหลด Dataset

### 1. ติดตั้ง Dependencies
```bash
pip install -r requirements.txt
```

### 2. ตั้งค่า Kaggle API Token (Optional / แนะนำ)
คัดลอกไฟล์ `.env.example` เป็น `.env` และใส่ Kaggle Token:
```bash
cp .env.example .env
```
ภายใน `.env`:
```env
KAGGLE_API_TOKEN=your_kaggle_api_token_here
```

### 3. รัน Script ดาวน์โหลด Dataset
```bash
python download_data.py
```
> ข้อมูลจะถูกดาวน์โหลดและนำมาจัดเก็บไว้ที่โฟลเดอร์ `data/raw` ซึ่งถูกระบุไว้ใน `.gitignore` เพื่อไม่ให้ไฟล์ภาพและ labels ถูก push ขึ้น GitHub repository
