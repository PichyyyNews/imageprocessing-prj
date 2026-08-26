# ✂️ รายงานการแบ่งชุดข้อมูล (Data Splitting Report)
**Methodology:** Multi-Label Stratified Holdout Splitting (Greedy Stratification)  
**Target Ratios:** Train (70%) / Val (15%) / Test (15%)  
**Random Seed:** `42` (100% Deterministic & Reproducible)  

---

## 1. การประเมินชุดข้อมูลเดิมและเหตุผลความจำเป็นในการแบ่งใหม่ (Motivation & Critique)
จากการตรวจสอบชุดข้อมูลดิบดั้งเดิมจาก Kaggle พบข้อบกพร่องทางสถิติที่สำคัญดังนี้:
1. **ไม่มีชุดทดสอบ (Missing Test Set):** มีเพียง `train` (90%) และ `valid` (10%) โดยไม่มี Held-out Test Set อิสระ ทำให้ไม่สามารถประเมิน Unbiased Generalization Performance ได้อย่างแท้จริง
2. **การขาดหายของคลาสใน Validation Set เดิม (Zero-Representation Defect):** ในชุด Validation เดิม มีถึง **5 คลาส** ที่ไม่มีตัวอย่างปรากฏเลยแม้แต่ภาพเดียว (`ambulance`, `army vehicle`, `auto rickshaw`, `garbagevan`, `human hauler`) ส่งผลให้ค่า Validation Metric (mAP) ไม่สามารถวัดประสิทธิภาพของคลาสเหล่านี้ได้

### 🎯 การแก้ไขและหลักการแบ่งชุดข้อมูลใหม่:
- นำเทคนิค **Multi-Label Stratified Sampling** มาใช้ เพื่อจัดสรรให้ทุกคลาส (รวมถึงคลาสที่มีตัวอย่างน้อย เช่น `garbagevan`) กระจายตัวอย่างทั่วถึงในทุก Split
- แบ่งสัดส่วนมาตรฐาน $70\% / 15\% / 15\%$ รองรับการ Train, Validate (Hyperparameter Tuning), และ Test (Final Evaluation)

---

## 2. สรุปผลการแบ่งชุดข้อมูล (Splitting Overview)
| Split | จำนวนภาพ (Images) | สัดส่วนจริง (%) | จำนวนวัตถุ (BBoxes) | จำนวนคลาสที่ครอบคลุม | คลาสที่ขาดหาย (Missing) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TRAIN** | 2,098 ภาพ | 69.84% | 16,981 boxes | 21 / 21 คลาส | 0 คลาส |
| **VAL** | 456 ภาพ | 15.18% | 3,671 boxes | 21 / 21 คลาส | 0 คลาส |
| **TEST** | 450 ภาพ | 14.98% | 3,672 boxes | 21 / 21 คลาส | 0 คลาส |

---

## 3. การกระจายตัวของทั้ง 21 คลาสในแต่ละ Split (Detailed Class Distribution Matrix)
ตารางแสดงการกระจายตัวของจำนวนวัตถุในแต่ละคลาสเปรียบเทียบข้าม Train, Validation, และ Test:

| Class ID | Class Name | Train Count | Val Count | Test Count | Total Count | Train % | Val % | Test % |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | `ambulance` | 51 | 10 | 9 | **70** | 72.9% | 14.3% | 12.9% |
| 1 | `army vehicle` | 30 | 6 | 7 | **43** | 69.8% | 14.0% | 16.3% |
| 2 | `auto rickshaw` | 255 | 54 | 63 | **372** | 68.5% | 14.5% | 16.9% |
| 3 | `bicycle` | 316 | 71 | 72 | **459** | 68.8% | 15.5% | 15.7% |
| 4 | `bus` | 2,325 | 500 | 507 | **3,332** | 69.8% | 15.0% | 15.2% |
| 5 | `car` | 3,834 | 815 | 825 | **5,474** | 70.0% | 14.9% | 15.1% |
| 6 | `garbagevan` | 1 | 1 | 1 | **3** | 33.3% | 33.3% | 33.3% |
| 7 | `human hauler` | 116 | 27 | 26 | **169** | 68.6% | 16.0% | 15.4% |
| 8 | `minibus` | 65 | 16 | 14 | **95** | 68.4% | 16.8% | 14.7% |
| 9 | `minivan` | 649 | 140 | 142 | **931** | 69.7% | 15.0% | 15.3% |
| 10 | `motorbike` | 1,595 | 348 | 341 | **2,284** | 69.8% | 15.2% | 14.9% |
| 11 | `pickup` | 846 | 190 | 189 | **1,225** | 69.1% | 15.5% | 15.4% |
| 12 | `policecar` | 21 | 6 | 5 | **32** | 65.6% | 18.8% | 15.6% |
| 13 | `rickshaw` | 2,469 | 525 | 532 | **3,526** | 70.0% | 14.9% | 15.1% |
| 14 | `scooter` | 27 | 6 | 5 | **38** | 71.1% | 15.8% | 13.2% |
| 15 | `suv` | 602 | 127 | 128 | **857** | 70.2% | 14.8% | 14.9% |
| 16 | `taxi` | 43 | 8 | 9 | **60** | 71.7% | 13.3% | 15.0% |
| 17 | `three wheelers -CNG-` | 2,085 | 456 | 444 | **2,985** | 69.8% | 15.3% | 14.9% |
| 18 | `truck` | 1,041 | 232 | 222 | **1,495** | 69.6% | 15.5% | 14.8% |
| 19 | `van` | 529 | 114 | 112 | **755** | 70.1% | 15.1% | 14.8% |
| 20 | `wheelbarrow` | 81 | 19 | 19 | **119** | 68.1% | 16.0% | 16.0% |

---

## 4. การป้องกัน Data Leakage และความสามารถในการทำซ้ำ (Leakage Prevention & Reproducibility)
- **No Overlap (Disjoint Subsets):** ตรวจสอบและยืนยันว่าไม่มีการซ้ำซ้อนของรูปภาพใดๆ ข้ามระหว่างชุด Train, Val และ Test ($Train \cap Val \cap Test = \emptyset$)
- **Fixed Random Seed:** ใช้ Seed `42` ทำให้การสุ่มแบ่งข้อมูลได้ผลลัพธ์เหมือนเดิมทุกครั้งที่รัน (Deterministic Behavior)
- **YOLO Config Generation:** สร้างไฟล์ `data/split/dataset_split.yaml` สำหรับสั่ง Train ในโมเดล Object Detection ได้ทันที

---

*รายงานนี้จัดทำขึ้นโดยอัตโนมัติผ่านโมดูล `src/data/split_data.py`*