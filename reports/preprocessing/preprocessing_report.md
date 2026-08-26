# 🖼️ รายงานการเตรียมข้อมูลและการประมวลผลรูปภาพ (Image Preprocessing Report)
**Target Resolution:** `640 x 640` (Aspect-Ratio Preserved Letterbox)  
**Padding Color:** RGB `(114, 114, 114)` (Neutral Gray)  
**CLAHE Enhancement:** `Disabled (Default Baseline)`  

---

## 1. วัตถุประสงค์และเหตุผลเชิงเทคนิค (Objectives & Rationale)
จากผลการวิเคราะห์ EDA พบว่ารูปภาพมีขนาด Resolution หลากหลาย ($360 \times 148$ ถึง $1125 \times 640$) และมีสัดส่วนแบบ Landscape เป็นส่วนใหญ่ (82.7%) การปรับขนาดด้วยวิธี Direct Resize (ยืดหรือบีบภาพ) จะทำให้รูปร่างของยานพาหนะบิดเบี้ยว (Distortion) ส่งผลเสียต่อการเรียนรู้ฟีเจอร์ของโครงข่ายประสาทเทียม

### 🎯 กลยุทธ์การ Preprocessing ที่นำมาใช้:
1. **Letterbox Padding Resizing:** ย่อ/ขยายภาพโดยคงอัตราส่วนเดิม ($Aspect\ Ratio$) และเติม Padding ด้วยสีเทามาตรฐาน $(114, 114, 114)$ ให้อยู่ในขนาดมาตรฐาน $640 \times 640$
2. **Bounding Box Recalculation:** คำนวณพิกัด Bounding Box ใหม่ตาม Scaling Factor ($r$) และ Offset Padding ($pad_x, pad_y$) อย่างแม่นยำ 100%
3. **Coordinate Boundary Clipping:** ป้องกันค่าพิกัดล้นขอบภาพโดยควบคุมให้อยู่ในช่วง $[0.0, 1.0]$
4. **Degenerate Box Filtering:** กรอง Bounding Box ที่มีขนาดเล็กเกินกว่าพิกเซลจริง ($width, height \le 0.001$) ออกเพื่อป้องกัน Noise ในการฝึกสอน

---

## 2. สูตรคณิตศาสตร์การแปลงพิกัด (Mathematical Transformation Formulas)
### 2.1 สเกลและขนาดใหม่ของภาพ (Scaling & Dimensions)
$$r = \min\left(\frac{W_{target}}{W_{orig}}, \frac{H_{target}}{H_{orig}}\right)$$
$$W_{new} = \text{round}(W_{orig} \times r), \quad H_{new} = \text{round}(H_{orig} \times r)$$
$$pad_x = \left\lfloor \frac{W_{target} - W_{new}}{2} \right\rfloor, \quad pad_y = \left\lfloor \frac{H_{target} - H_{new}}{2} \right\rfloor$$

### 2.2 การแปลงพิกัด YOLO Bounding Box ($[x_c, y_c, w, h] \rightarrow [x_c', y_c', w', h']$)
$$x_c' = \frac{(x_c \times W_{orig}) \times r + pad_x}{W_{target}}, \quad y_c' = \frac{(y_c \times H_{orig}) \times r + pad_y}{H_{target}}$$
$$w' = \frac{(w \times W_{orig}) \times r}{W_{target}}, \quad h' = \frac{(h \times H_{orig}) \times r}{H_{target}}$$

---

## 3. ผลลัพธ์การประมวลผล (Preprocessing Results Summary)
| Split | จำนวนภาพที่ประมวลผล | จำนวน Bounding Box ทั้งหมด | Scale Factor เฉลี่ย |
| :--- | :---: | :---: | :---: |
| **train** | 2,704 ภาพ | 21,764 boxes | 1.0000 |
| **valid** | 300 ภาพ | 2,560 boxes | 0.9971 |

- **รวมรูปภาพที่ผ่านการ Preprocess ทั้งหมด:** **3,004** ภาพ
- **รวม Bounding Boxes ที่ผ่านการแปลงพิกัด:** **24,324** boxes
- **ขนาดไฟล์หลัง Preprocess:** ทุกภาพมีขนาด $640 \times 640 \times 3$ สม่ำเสมอ 100%

---

## 4. ความพร้อมสำหรับการนำไปใช้งาน (Production Readiness Checklist)
- [x] รูปภาพทั้งหมดมีมิติขนาด $640 \times 640$ สม่ำเสมอ พร้อมส่งเข้า Batch Training
- [x] พิกัด Bounding Box ได้รับการปรับตามสเกล Letterbox ถูกต้อง 100%
- [x] โครงสร้างไฟล์และ Config (`data_processed.yaml`) ตรงตามมาตรฐาน YOLOv8 / YOLOv11 / RT-DETR
- [x] ไฟล์ภาพและ Label ถูกจัดเก็บไว้ใน `data/processed/` และถูกกันไม่ให้ push ขึ้น Git ผ่าน `.gitignore`

---

*รายงานนี้จัดทำขึ้นโดยอัตโนมัติผ่านโมดูล `src/preprocessing/preprocessor.py`*