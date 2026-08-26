# 📊 รายงานการสำรวจและวิเคราะห์ข้อมูลเชิงลึก (Comprehensive Exploratory Data Analysis Report)
**Dataset Name:** Road Vehicle Images Dataset (`ashfakyeafi/road-vehicle-images-dataset`)  
**Format:** YOLO Object Detection (`data_1.yaml`, normalized `[class, x_center, y_center, width, height]`)  
**Dataset Location:** `C:\Users\Newsk\OneDrive\Desktop\imageprocessing-prj\data\raw`  

---

## 1. Executive Summary (บทสรุปภาพรวม)
- **จำนวนรูปภาพทั้งหมด:** **3,004** ภาพ (Train: 2,704 ภาพ, Validation: 300 ภาพ)
- **จำนวนวัตถุ (Bounding Boxes) ทั้งหมด:** **24,348** วัตถุ
- **จำนวนคลาสทั้งหมด:** **21** คลาส (ประเภทยานพาหนะทางถนน)
- **ค่าเฉลี่ยจำนวนวัตถุต่อภาพ:** **8.11** วัตถุ/ภาพ (สูงสุด 122 วัตถุ)
- **อัตราความไม่สมดุลของข้อมูล (Class Imbalance Ratio):** **1824.67 : 1** (ความถี่คลาสสูงสุดเทียบกับคลาสต้อยต่ำสุด)
- **สถานะความสมบูรณ์ของไฟล์ (Data Integrity):** ตรวจพบภาพเสียหาย 0 ไฟล์, Bounding Box ผิดปกติ 24 รายการ

---

## 2. โครงสร้างชุดข้อมูลและความสมบูรณ์ของไฟล์ (Dataset Integrity & File Verification)
| Split | Image Count | Label Count | Missing Labels | Missing Images | Total BBoxes | Avg BBoxes/Img | Avg File Size (KB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **train** | 2,704 | 2,704 | 0 | 0 | 21,764 | 8.05 | 38.9 KB |
| **valid** | 300 | 300 | 0 | 0 | 2,584 | 8.61 | 34.6 KB |

- **ขนาดรวมของรูปภาพทั้งหมด (Disk Storage):** ~112.90 MB
- **ขนาดไฟล์ภาพ:** ต่ำสุด 8.0 KB, สูงสุด 114.6 KB, ค่าเฉลี่ย 38.5 KB

---

## 3. การวิเคราะห์คุณลักษณะทางกายภาพของรูปภาพ (Image Physical Characteristics)
### 3.1 ขนาดและความละเอียดของรูปภาพ (Image Dimensions & Resolutions)
- **ความกว้าง (Width):** ต่ำสุด `360px`, สูงสุด `1125px`, ค่าเฉลี่ย `602.7px` (Std: `88.9`)
- **ความสูง (Height):** ต่ำสุด `148px`, สูงสุด `640px`, ค่าเฉลี่ย `411.2px` (Std: `111.0`)

**ความละเอียดยอดนิยม (Top 5 Resolutions):**
| ลำดับ | Resolution (W x H) | จำนวนภาพ | สัดส่วน (%) |
| :---: | :---: | :---: | :---: |
| 1 | `640 x 360` | 1,672 | 55.66% |
| 2 | `360 x 640` | 274 | 9.12% |
| 3 | `640 x 359` | 268 | 8.92% |
| 4 | `480 x 640` | 138 | 4.59% |
| 5 | `640 x 362` | 85 | 2.83% |

### 3.2 สัดส่วนภาพ (Aspect Ratio Distribution)
- **ค่าเฉลี่ย Aspect Ratio ($W/H$):** `1.589` (Median: `1.778`)
- **แนวนอน (Landscape - $AR > 1.05$):** **2,483** ภาพ (82.7%)
- **จัตุรัส (Square - $0.95 \le AR \le 1.05$):** **21** ภาพ (0.7%)
- **แนวตั้ง (Portrait - $AR < 0.95$):** **500** ภาพ (16.6%)

### 3.3 สถิติค่าพิกเซลและระดับความสว่าง (Color & Pixel Statistics)
| Channel / Metric | ค่าเฉลี่ย (Mean) | ค่าเบี่ยงเบนมาตรฐาน (Std) |
| :--- | :---: | :---: |
| **Red (R)** | 120.60 / 255 | 60.88 |
| **Green (G)** | 122.31 / 255 | 62.02 |
| **Blue (B)** | 120.52 / 255 | 63.86 |
| **Luminance (ความสว่าง)** | 121.60 / 255 | 26.30 |
| **Contrast (ความต่างระดับสี)** | 61.04 | - |

---

## 4. การกระจายตัวของคลาสและความไม่สมดุล (Class Distribution & Imbalance Analysis)
ชุดข้อมูลประกอบด้วย 21 คลาสของยานพาหนะ ตารางด้านล่างแสดงการกระจายตัวของแต่ละคลาสพร้อมสัดส่วน:

| Class ID | Class Name | Train Count | Valid Count | Total Count | Proportion (%) | Density Ranking |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 5 | `car` | 4,632 | 842 | **5,474** | 22.48% | #1 |
| 13 | `rickshaw` | 3,336 | 203 | **3,539** | 14.54% | #2 |
| 4 | `bus` | 2,909 | 430 | **3,339** | 13.71% | #3 |
| 17 | `three wheelers -CNG-` | 2,734 | 252 | **2,986** | 12.26% | #4 |
| 10 | `motorbike` | 1,949 | 335 | **2,284** | 9.38% | #5 |
| 18 | `truck` | 1,411 | 84 | **1,495** | 6.14% | #6 |
| 11 | `pickup` | 1,083 | 142 | **1,225** | 5.03% | #7 |
| 9 | `minivan` | 822 | 110 | **932** | 3.83% | #8 |
| 15 | `suv` | 798 | 60 | **858** | 3.52% | #9 |
| 19 | `van` | 693 | 62 | **755** | 3.10% | #10 |
| 3 | `bicycle` | 427 | 32 | **459** | 1.89% | #11 |
| 2 | `auto rickshaw` | 372 | 0 | **372** | 1.53% | #12 |
| 7 | `human hauler` | 169 | 0 | **169** | 0.69% | #13 |
| 20 | `wheelbarrow` | 111 | 9 | **120** | 0.49% | #14 |
| 8 | `minibus` | 93 | 2 | **95** | 0.39% | #15 |
| 0 | `ambulance` | 70 | 0 | **70** | 0.29% | #16 |
| 16 | `taxi` | 41 | 19 | **60** | 0.25% | #17 |
| 1 | `army vehicle` | 43 | 0 | **43** | 0.18% | #18 |
| 14 | `scooter` | 37 | 1 | **38** | 0.16% | #19 |
| 12 | `policecar` | 31 | 1 | **32** | 0.13% | #20 |
| 6 | `garbagevan` | 3 | 0 | **3** | 0.01% | #21 |

### 4.1 การประเมิน Class Imbalance
- **คลาสที่มีตัวอย่างมากที่สุด (Dominant Class):** `car` (5,474 instances)
- **คลาสที่มีตัวอย่างน้อยที่สุด (Rare Class):** `garbagevan` (3 instances)
- **Imbalance Ratio:** `1824.67x`

> [!NOTE]
> คลาสส่วนใหญ่ที่มีความถี่สูง เช่น `car`, `bus`, `auto rickshaw` เป็นยานพาหนะทั่วไปบนท้องถนน ในขณะที่คลาสเฉพาะทาง เช่น `wheelbarrow`, `garbagevan`, `policecar` มีสัดส่วนน้อยกว่า ซึ่งสอดคล้องกับธรรมชาติของข้อมูลจราจรจริง

---

## 5. การวิเคราะห์รูปร่างและตำแหน่งของ Bounding Box (Bounding Box Morphology & Spatial Distribution)
### 5.1 ความหนาแน่นของวัตถุต่อภาพ (Object Density per Image)
| จำนวนวัตถุต่อภาพ | จำนวนภาพ | สัดส่วน (%) |
| :--- | :---: | :---: |
| 0 วัตถุ (Background/Empty) | 2 | 0.07% |
| 1 วัตถุ (Single Object) | 209 | 6.96% |
| 2 - 5 วัตถุ (Low-to-Medium Density) | 1,247 | 41.51% |
| 6 - 10 วัตถุ (Dense Scene) | 1,002 | 33.36% |
| > 10 วัตถุ (Crowded Traffic) | 544 | 18.11% |

### 5.2 การจำแนกขนาดวัตถุ (Object Scale Breakdown - Normalized Area)
- **วัตถุขนาดเล็ก (Small Objects, Area $< 5\%$ ของภาพ):** **88.34%**
- **วัตถุขนาดกลาง (Medium Objects, $5\% \le Area \le 25\%$):** **9.57%**
- **วัตถุขนาดใหญ่ (Large Objects, Area $> 25\%$):** **2.09%**
- **ค่าเฉลี่ยพื้นที่ Bounding Box:** `0.0265` (Median: `0.0047`)

### 5.3 ตำแหน่งเชิงพื้นที่ (Spatial Center Distribution)
- **จุดกึ่งกลางเฉลี่ยแกน X ($X_{center}$):** `0.4912` (ใกล้เคียงศูนย์กลางภาพ 0.50)
- **จุดกึ่งกลางเฉลี่ยแกน Y ($Y_{center}$):** `0.5059` (ค่อนไปทางครึ่งล่างของภาพ ซึ่งสอดคล้องกับระนาบถนนของการจราจร)

---

## 6. ข้อค้นพบสำคัญและข้อเสนอแนะสำหรับการ Preprocessing และ Modeling
### 💡 ข้อค้นพบหลัก (Key Insights)
1. **ความหลากหลายของขนาดภาพ:** รูปภาพมีขนาด Resolution แตกต่างกัน จึงจำเป็นต้องทำการ Resize แบบคงอัตราส่วน (Aspect Ratio Preserved / Letterbox Padding) เพื่อป้องกันภาพบิดเบี้ยว
2. **ความไม่สมดุลของคลาส (Class Imbalance):** มีคลาสเด่น (Dominant) และคลาสย่อย (Minority) จึงควรใช้เทคนิค Class-weighted Loss (เช่น Focal Loss) หรือ Data Augmentation สำหรับคลาสย่อย
3. **การกระจายตัวของขนาดวัตถุ:** มีทั้งวัตถุขนาดเล็ก (รถที่อยู่ไกล) และขนาดใหญ่ (รถที่อยู่ใกล้) ควรใช้โมเดลที่มี Multi-scale Feature Pyramid (เช่น YOLOv8, Feature Pyramid Network - FPN)
4. **ความสมบูรณ์ของพิกัด Annotation:** พิกัดทั้งหมดอยู่ในช่วงมาตรฐาน $[0, 1]$ รูปภาพสามารถอ่านได้ 100% ไม่มีไฟล์เสียหาย

### 🛠️ ขั้นตอนการ Preprocessing ที่แนะนำ
- **Letterbox Resize:** ปรับขนาดภาพเป็นมาตรฐาน (เช่น $640 \times 640$) พร้อม Padding สีเทา (114, 114, 114) และปรับพิกัด Bounding Box ตามสเกลใหม่อย่างแม่นยำ
- **Color Standardization & Normalization:** สเกลค่าพิกเซลให้อยู่ในช่วง $[0.0, 1.0]$ หรือมาตรฐาน ImageNet Mean/Std
- **Data Augmentation:** ใช้ Mosaic, Mixup, Random Horizontal Flip, HSV Color Jittering เพื่อเพิ่มความคงทนของโมเดล

---

*รายงานนี้จัดทำขึ้นโดยอัตโนมัติผ่านโมดูล `src/eda/eda_analyzer.py`*