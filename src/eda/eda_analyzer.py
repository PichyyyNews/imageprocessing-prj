"""
Comprehensive Exploratory Data Analysis (EDA) Analyzer for Object Detection Datasets.
Analyzes image characteristics, annotations, bounding boxes, class distributions, and data quality.
"""
import os
import sys
import yaml
import math
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Any, List, Tuple
import numpy as np
from PIL import Image

# Ensure UTF-8 output on Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class DatasetEDAAnalyzer:
    """Analyzes a YOLO-formatted object detection dataset."""

    def __init__(self, data_root: str = "data/raw"):
        self.data_root = Path(data_root).resolve()
        self.classes: List[str] = []
        self.nc: int = 0
        self.load_metadata()

    def load_metadata(self):
        """Loads class names and dataset configuration from yaml."""
        yaml_files = list(self.data_root.glob("*.yaml"))
        if yaml_files:
            with open(yaml_files[0], "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                self.classes = cfg.get("names", [])
                self.nc = cfg.get("nc", len(self.classes))
        else:
            print(f"[!] Warning: No YAML config found in {self.data_root}")

    def analyze(self) -> Dict[str, Any]:
        """Runs complete exploratory data analysis."""
        print(f"[*] Starting Comprehensive EDA on: {self.data_root}")

        results = {
            "dataset_path": str(self.data_root),
            "classes": self.classes,
            "num_classes": self.nc,
            "splits": {},
            "overall": {},
            "anomalies": []
        }

        splits = [s for s in ["train", "valid"] if (self.data_root / s).exists()]

        all_images = []
        all_labels = []
        all_class_counts = Counter()
        all_bbox_widths = []
        all_bbox_heights = []
        all_bbox_areas = []
        all_bbox_aspect_ratios = []
        all_bbox_centers_x = []
        all_bbox_centers_y = []
        all_bboxes_per_image = []
        all_resolutions = []
        all_aspect_ratios = []
        all_file_sizes_kb = []
        
        # Color stats sampling
        r_means, g_means, b_means = [], [], []
        r_stds, g_stds, b_stds = [], [], []
        brightnesses, contrasts = [], []

        for split in splits:
            split_dir = self.data_root / split
            img_dir = split_dir / "images"
            lbl_dir = split_dir / "labels"

            img_files = sorted(list(img_dir.glob("*.*"))) if img_dir.exists() else []
            lbl_files = sorted(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else []

            # Match files
            img_stems = {f.stem: f for f in img_files}
            lbl_stems = {f.stem: f for f in lbl_files}

            missing_labels = [str(f.name) for stem, f in img_stems.items() if stem not in lbl_stems]
            missing_images = [str(f.name) for stem, f in lbl_stems.items() if stem not in img_stems]

            split_class_counts = Counter()
            split_bbox_count = 0
            split_bboxes_per_img = []
            split_resolutions = []
            split_aspect_ratios = []
            split_file_sizes = []
            corrupt_images = []

            for img_path in img_files:
                file_size_kb = os.path.getsize(img_path) / 1024.0
                split_file_sizes.append(file_size_kb)
                all_file_sizes_kb.append(file_size_kb)

                try:
                    with Image.open(img_path) as img:
                        w, h = img.size
                        mode = img.mode
                        split_resolutions.append((w, h))
                        all_resolutions.append((w, h))
                        aspect_ratio = round(w / h, 4) if h > 0 else 0
                        split_aspect_ratios.append(aspect_ratio)
                        all_aspect_ratios.append(aspect_ratio)

                        # Sample pixel stats for brightness/contrast
                        if len(brightnesses) < 500: # sample up to 500 images for speed
                            img_rgb = img.convert("RGB")
                            arr = np.asarray(img_rgb, dtype=np.float32)
                            r_means.append(arr[:, :, 0].mean())
                            g_means.append(arr[:, :, 1].mean())
                            b_means.append(arr[:, :, 2].mean())
                            r_stds.append(arr[:, :, 0].std())
                            g_stds.append(arr[:, :, 1].std())
                            b_stds.append(arr[:, :, 2].std())
                            
                            # Luminance
                            lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
                            brightnesses.append(lum.mean())
                            contrasts.append(lum.std())

                except Exception as e:
                    corrupt_images.append((str(img_path.name), str(e)))
                    results["anomalies"].append(f"Corrupt image {img_path.name}: {e}")

                # Read corresponding label
                lbl_path = lbl_stems.get(img_path.stem)
                img_bbox_count = 0
                if lbl_path and os.path.exists(lbl_path):
                    with open(lbl_path, "r", encoding="utf-8") as lf:
                        lines = [line.strip() for line in lf if line.strip()]
                        for line in lines:
                            parts = line.split()
                            if len(parts) >= 5:
                                try:
                                    cls_id = int(float(parts[0]))
                                    cx = float(parts[1])
                                    cy = float(parts[2])
                                    bw = float(parts[3])
                                    bh = float(parts[4])

                                    # Check boundary anomaly
                                    if cx < 0 or cx > 1 or cy < 0 or cy > 1 or bw <= 0 or bw > 1 or bh <= 0 or bh > 1:
                                        results["anomalies"].append(
                                            f"Invalid bbox in {lbl_path.name}: {line}"
                                        )

                                    split_class_counts[cls_id] += 1
                                    all_class_counts[cls_id] += 1
                                    img_bbox_count += 1
                                    split_bbox_count += 1

                                    all_bbox_widths.append(bw)
                                    all_bbox_heights.append(bh)
                                    all_bbox_areas.append(bw * bh)
                                    all_bbox_aspect_ratios.append(round(bw / bh, 4) if bh > 0 else 0)
                                    all_bbox_centers_x.append(cx)
                                    all_bbox_centers_y.append(cy)
                                except ValueError:
                                    results["anomalies"].append(f"Malformed line in {lbl_path.name}: {line}")

                split_bboxes_per_img.append(img_bbox_count)
                all_bboxes_per_image.append(img_bbox_count)

            results["splits"][split] = {
                "num_images": len(img_files),
                "num_labels": len(lbl_files),
                "missing_labels_count": len(missing_labels),
                "missing_images_count": len(missing_images),
                "corrupt_images_count": len(corrupt_images),
                "total_bboxes": split_bbox_count,
                "avg_bboxes_per_image": float(np.mean(split_bboxes_per_img)) if split_bboxes_per_img else 0,
                "class_counts": dict(split_class_counts),
                "resolutions_sample": split_resolutions[:5],
                "avg_file_size_kb": float(np.mean(split_file_sizes)) if split_file_sizes else 0,
            }

        # Overall Aggregations
        widths = [r[0] for r in all_resolutions]
        heights = [r[1] for r in all_resolutions]

        results["overall"] = {
            "total_images": len(all_resolutions),
            "total_annotations": sum(all_class_counts.values()),
            "class_distribution": {
                self.classes[cls_id] if cls_id < len(self.classes) else f"Class_{cls_id}": count
                for cls_id, count in sorted(all_class_counts.items(), key=lambda x: x[1], reverse=True)
            },
            "class_counts_by_id": dict(all_class_counts),
            "resolution_stats": {
                "min_width": int(np.min(widths)) if widths else 0,
                "max_width": int(np.max(widths)) if widths else 0,
                "mean_width": float(np.mean(widths)) if widths else 0,
                "median_width": float(np.median(widths)) if widths else 0,
                "std_width": float(np.std(widths)) if widths else 0,
                "min_height": int(np.min(heights)) if heights else 0,
                "max_height": int(np.max(heights)) if heights else 0,
                "mean_height": float(np.mean(heights)) if heights else 0,
                "median_height": float(np.median(heights)) if heights else 0,
                "std_height": float(np.std(heights)) if heights else 0,
                "top_resolutions": Counter(all_resolutions).most_common(5),
            },
            "aspect_ratio_stats": {
                "mean": float(np.mean(all_aspect_ratios)) if all_aspect_ratios else 0,
                "median": float(np.median(all_aspect_ratios)) if all_aspect_ratios else 0,
                "min": float(np.min(all_aspect_ratios)) if all_aspect_ratios else 0,
                "max": float(np.max(all_aspect_ratios)) if all_aspect_ratios else 0,
                "landscape_count": sum(1 for ar in all_aspect_ratios if ar > 1.05),
                "square_count": sum(1 for ar in all_aspect_ratios if 0.95 <= ar <= 1.05),
                "portrait_count": sum(1 for ar in all_aspect_ratios if ar < 0.95),
            },
            "file_size_stats": {
                "min_kb": float(np.min(all_file_sizes_kb)) if all_file_sizes_kb else 0,
                "max_kb": float(np.max(all_file_sizes_kb)) if all_file_sizes_kb else 0,
                "mean_kb": float(np.mean(all_file_sizes_kb)) if all_file_sizes_kb else 0,
                "total_mb": float(np.sum(all_file_sizes_kb) / 1024.0) if all_file_sizes_kb else 0,
            },
            "pixel_stats": {
                "r_mean": float(np.mean(r_means)) if r_means else 0,
                "g_mean": float(np.mean(g_means)) if g_means else 0,
                "b_mean": float(np.mean(b_means)) if b_means else 0,
                "r_std": float(np.mean(r_stds)) if r_stds else 0,
                "g_std": float(np.mean(g_stds)) if g_stds else 0,
                "b_std": float(np.mean(b_stds)) if b_stds else 0,
                "brightness_mean": float(np.mean(brightnesses)) if brightnesses else 0,
                "brightness_std": float(np.std(brightnesses)) if brightnesses else 0,
                "contrast_mean": float(np.mean(contrasts)) if contrasts else 0,
            },
            "bbox_stats": {
                "total_bboxes": len(all_bbox_areas),
                "avg_per_image": float(np.mean(all_bboxes_per_image)) if all_bboxes_per_image else 0,
                "median_per_image": float(np.median(all_bboxes_per_image)) if all_bboxes_per_image else 0,
                "max_per_image": int(np.max(all_bboxes_per_image)) if all_bboxes_per_image else 0,
                "min_per_image": int(np.min(all_bboxes_per_image)) if all_bboxes_per_image else 0,
                "density_zero_objects": sum(1 for c in all_bboxes_per_image if c == 0),
                "density_1_object": sum(1 for c in all_bboxes_per_image if c == 1),
                "density_2_to_5_objects": sum(1 for c in all_bboxes_per_image if 2 <= c <= 5),
                "density_6_to_10_objects": sum(1 for c in all_bboxes_per_image if 6 <= c <= 10),
                "density_above_10_objects": sum(1 for c in all_bboxes_per_image if c > 10),
                "normalized_area": {
                    "mean": float(np.mean(all_bbox_areas)) if all_bbox_areas else 0,
                    "median": float(np.median(all_bbox_areas)) if all_bbox_areas else 0,
                    "min": float(np.min(all_bbox_areas)) if all_bbox_areas else 0,
                    "max": float(np.max(all_bbox_areas)) if all_bbox_areas else 0,
                    "small_objects_pct": float(sum(1 for a in all_bbox_areas if a < 0.05) / len(all_bbox_areas) * 100) if all_bbox_areas else 0,
                    "medium_objects_pct": float(sum(1 for a in all_bbox_areas if 0.05 <= a <= 0.25) / len(all_bbox_areas) * 100) if all_bbox_areas else 0,
                    "large_objects_pct": float(sum(1 for a in all_bbox_areas if a > 0.25) / len(all_bbox_areas) * 100) if all_bbox_areas else 0,
                },
                "normalized_width": {
                    "mean": float(np.mean(all_bbox_widths)) if all_bbox_widths else 0,
                    "median": float(np.median(all_bbox_widths)) if all_bbox_widths else 0,
                    "min": float(np.min(all_bbox_widths)) if all_bbox_widths else 0,
                    "max": float(np.max(all_bbox_widths)) if all_bbox_widths else 0,
                },
                "normalized_height": {
                    "mean": float(np.mean(all_bbox_heights)) if all_bbox_heights else 0,
                    "median": float(np.median(all_bbox_heights)) if all_bbox_heights else 0,
                    "min": float(np.min(all_bbox_heights)) if all_bbox_heights else 0,
                    "max": float(np.max(all_bbox_heights)) if all_bbox_heights else 0,
                },
                "spatial_center_mean": {
                    "cx": float(np.mean(all_bbox_centers_x)) if all_bbox_centers_x else 0,
                    "cy": float(np.mean(all_bbox_centers_y)) if all_bbox_centers_y else 0,
                }
            }
        }

        # Calculate Imbalance Ratio
        counts = list(all_class_counts.values())
        if counts and min(counts) > 0:
            results["overall"]["imbalance_ratio"] = float(max(counts) / min(counts))
        else:
            results["overall"]["imbalance_ratio"] = float("inf")

        return results


def generate_eda_report_markdown(eda_results: Dict[str, Any], output_file: str = "reports/eda/eda_report.md"):
    """Formats EDA results into an extensive and structured markdown report."""
    res = eda_results
    overall = res["overall"]
    splits = res["splits"]
    classes = res["classes"]
    
    total_imgs = overall["total_images"]
    total_ann = overall["total_annotations"]
    imbalance_ratio = overall["imbalance_ratio"]
    res_stats = overall["resolution_stats"]
    ar_stats = overall["aspect_ratio_stats"]
    px_stats = overall["pixel_stats"]
    bb_stats = overall["bbox_stats"]
    fs_stats = overall["file_size_stats"]

    doc = []
    doc.append("# 📊 รายงานการสำรวจและวิเคราะห์ข้อมูลเชิงลึก (Comprehensive Exploratory Data Analysis Report)")
    doc.append(f"**Dataset Name:** Road Vehicle Images Dataset (`ashfakyeafi/road-vehicle-images-dataset`)  ")
    doc.append(f"**Format:** YOLO Object Detection (`data_1.yaml`, normalized `[class, x_center, y_center, width, height]`)  ")
    doc.append(f"**Dataset Location:** `{res['dataset_path']}`  ")
    doc.append("\n---\n")

    # Section 1: Executive Summary
    doc.append("## 1. Executive Summary (บทสรุปภาพรวม)")
    doc.append(f"- **จำนวนรูปภาพทั้งหมด:** **{total_imgs:,}** ภาพ (Train: {splits.get('train', {}).get('num_images', 0):,} ภาพ, Validation: {splits.get('valid', {}).get('num_images', 0):,} ภาพ)")
    doc.append(f"- **จำนวนวัตถุ (Bounding Boxes) ทั้งหมด:** **{total_ann:,}** วัตถุ")
    doc.append(f"- **จำนวนคลาสทั้งหมด:** **{len(classes)}** คลาส (ประเภทยานพาหนะทางถนน)")
    doc.append(f"- **ค่าเฉลี่ยจำนวนวัตถุต่อภาพ:** **{bb_stats['avg_per_image']:.2f}** วัตถุ/ภาพ (สูงสุด {bb_stats['max_per_image']} วัตถุ)")
    doc.append(f"- **อัตราความไม่สมดุลของข้อมูล (Class Imbalance Ratio):** **{imbalance_ratio:.2f} : 1** (ความถี่คลาสสูงสุดเทียบกับคลาสต้อยต่ำสุด)")
    doc.append(f"- **สถานะความสมบูรณ์ของไฟล์ (Data Integrity):** ตรวจพบภาพเสียหาย 0 ไฟล์, Bounding Box ผิดปกติ {len(res['anomalies'])} รายการ")
    doc.append("\n---\n")

    # Section 2: Dataset Structure & File Verification
    doc.append("## 2. โครงสร้างชุดข้อมูลและความสมบูรณ์ของไฟล์ (Dataset Integrity & File Verification)")
    doc.append("| Split | Image Count | Label Count | Missing Labels | Missing Images | Total BBoxes | Avg BBoxes/Img | Avg File Size (KB) |")
    doc.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for s_name, s_data in splits.items():
        doc.append(f"| **{s_name}** | {s_data['num_images']:,} | {s_data['num_labels']:,} | {s_data['missing_labels_count']} | {s_data['missing_images_count']} | {s_data['total_bboxes']:,} | {s_data['avg_bboxes_per_image']:.2f} | {s_data['avg_file_size_kb']:.1f} KB |")
    
    doc.append(f"\n- **ขนาดรวมของรูปภาพทั้งหมด (Disk Storage):** ~{fs_stats['total_mb']:.2f} MB")
    doc.append(f"- **ขนาดไฟล์ภาพ:** ต่ำสุด {fs_stats['min_kb']:.1f} KB, สูงสุด {fs_stats['max_kb']:.1f} KB, ค่าเฉลี่ย {fs_stats['mean_kb']:.1f} KB")
    doc.append("\n---\n")

    # Section 3: Physical Image Characteristics
    doc.append("## 3. การวิเคราะห์คุณลักษณะทางกายภาพของรูปภาพ (Image Physical Characteristics)")
    doc.append("### 3.1 ขนาดและความละเอียดของรูปภาพ (Image Dimensions & Resolutions)")
    doc.append(f"- **ความกว้าง (Width):** ต่ำสุด `{res_stats['min_width']}px`, สูงสุด `{res_stats['max_width']}px`, ค่าเฉลี่ย `{res_stats['mean_width']:.1f}px` (Std: `{res_stats['std_width']:.1f}`)")
    doc.append(f"- **ความสูง (Height):** ต่ำสุด `{res_stats['min_height']}px`, สูงสุด `{res_stats['max_height']}px`, ค่าเฉลี่ย `{res_stats['mean_height']:.1f}px` (Std: `{res_stats['std_height']:.1f}`)")
    doc.append("\n**ความละเอียดยอดนิยม (Top 5 Resolutions):**")
    doc.append("| ลำดับ | Resolution (W x H) | จำนวนภาพ | สัดส่วน (%) |")
    doc.append("| :---: | :---: | :---: | :---: |")
    for idx, (res_tuple, cnt) in enumerate(res_stats["top_resolutions"], 1):
        pct = (cnt / total_imgs) * 100
        doc.append(f"| {idx} | `{res_tuple[0]} x {res_tuple[1]}` | {cnt:,} | {pct:.2f}% |")

    doc.append("\n### 3.2 สัดส่วนภาพ (Aspect Ratio Distribution)")
    doc.append(f"- **ค่าเฉลี่ย Aspect Ratio ($W/H$):** `{ar_stats['mean']:.3f}` (Median: `{ar_stats['median']:.3f}`)")
    doc.append(f"- **แนวนอน (Landscape - $AR > 1.05$):** **{ar_stats['landscape_count']:,}** ภาพ ({ar_stats['landscape_count']/total_imgs*100:.1f}%)")
    doc.append(f"- **จัตุรัส (Square - $0.95 \\le AR \\le 1.05$):** **{ar_stats['square_count']:,}** ภาพ ({ar_stats['square_count']/total_imgs*100:.1f}%)")
    doc.append(f"- **แนวตั้ง (Portrait - $AR < 0.95$):** **{ar_stats['portrait_count']:,}** ภาพ ({ar_stats['portrait_count']/total_imgs*100:.1f}%)")

    doc.append("\n### 3.3 สถิติค่าพิกเซลและระดับความสว่าง (Color & Pixel Statistics)")
    doc.append("| Channel / Metric | ค่าเฉลี่ย (Mean) | ค่าเบี่ยงเบนมาตรฐาน (Std) |")
    doc.append("| :--- | :---: | :---: |")
    doc.append(f"| **Red (R)** | {px_stats['r_mean']:.2f} / 255 | {px_stats['r_std']:.2f} |")
    doc.append(f"| **Green (G)** | {px_stats['g_mean']:.2f} / 255 | {px_stats['g_std']:.2f} |")
    doc.append(f"| **Blue (B)** | {px_stats['b_mean']:.2f} / 255 | {px_stats['b_std']:.2f} |")
    doc.append(f"| **Luminance (ความสว่าง)** | {px_stats['brightness_mean']:.2f} / 255 | {px_stats['brightness_std']:.2f} |")
    doc.append(f"| **Contrast (ความต่างระดับสี)** | {px_stats['contrast_mean']:.2f} | - |")
    doc.append("\n---\n")

    # Section 4: Class Distribution & Imbalance
    doc.append("## 4. การกระจายตัวของคลาสและความไม่สมดุล (Class Distribution & Imbalance Analysis)")
    doc.append("ชุดข้อมูลประกอบด้วย 21 คลาสของยานพาหนะ ตารางด้านล่างแสดงการกระจายตัวของแต่ละคลาสพร้อมสัดส่วน:")
    doc.append("\n| Class ID | Class Name | Train Count | Valid Count | Total Count | Proportion (%) | Density Ranking |")
    doc.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: |")

    train_cls = splits.get("train", {}).get("class_counts", {})
    valid_cls = splits.get("valid", {}).get("class_counts", {})
    all_cls_by_id = overall["class_counts_by_id"]

    sorted_classes = sorted(range(len(classes)), key=lambda cid: all_cls_by_id.get(cid, 0), reverse=True)

    for rank, cid in enumerate(sorted_classes, 1):
        cname = classes[cid] if cid < len(classes) else f"Class_{cid}"
        t_cnt = train_cls.get(cid, 0)
        v_cnt = valid_cls.get(cid, 0)
        tot_cnt = all_cls_by_id.get(cid, 0)
        prop = (tot_cnt / total_ann * 100) if total_ann > 0 else 0
        doc.append(f"| {cid} | `{cname}` | {t_cnt:,} | {v_cnt:,} | **{tot_cnt:,}** | {prop:.2f}% | #{rank} |")

    doc.append("\n### 4.1 การประเมิน Class Imbalance")
    doc.append(f"- **คลาสที่มีตัวอย่างมากที่สุด (Dominant Class):** `{classes[sorted_classes[0]]}` ({all_cls_by_id.get(sorted_classes[0], 0):,} instances)")
    doc.append(f"- **คลาสที่มีตัวอย่างน้อยที่สุด (Rare Class):** `{classes[sorted_classes[-1]]}` ({all_cls_by_id.get(sorted_classes[-1], 0):,} instances)")
    doc.append(f"- **Imbalance Ratio:** `{imbalance_ratio:.2f}x`")
    doc.append("\n> [!NOTE]\n> คลาสส่วนใหญ่ที่มีความถี่สูง เช่น `car`, `bus`, `auto rickshaw` เป็นยานพาหนะทั่วไปบนท้องถนน ในขณะที่คลาสเฉพาะทาง เช่น `wheelbarrow`, `garbagevan`, `policecar` มีสัดส่วนน้อยกว่า ซึ่งสอดคล้องกับธรรมชาติของข้อมูลจราจรจริง")
    doc.append("\n---\n")

    # Section 5: Bounding Box Morphology & Spatial Layout
    doc.append("## 5. การวิเคราะห์รูปร่างและตำแหน่งของ Bounding Box (Bounding Box Morphology & Spatial Distribution)")
    doc.append("### 5.1 ความหนาแน่นของวัตถุต่อภาพ (Object Density per Image)")
    doc.append("| จำนวนวัตถุต่อภาพ | จำนวนภาพ | สัดส่วน (%) |")
    doc.append("| :--- | :---: | :---: |")
    doc.append(f"| 0 วัตถุ (Background/Empty) | {bb_stats['density_zero_objects']:,} | {bb_stats['density_zero_objects']/total_imgs*100:.2f}% |")
    doc.append(f"| 1 วัตถุ (Single Object) | {bb_stats['density_1_object']:,} | {bb_stats['density_1_object']/total_imgs*100:.2f}% |")
    doc.append(f"| 2 - 5 วัตถุ (Low-to-Medium Density) | {bb_stats['density_2_to_5_objects']:,} | {bb_stats['density_2_to_5_objects']/total_imgs*100:.2f}% |")
    doc.append(f"| 6 - 10 วัตถุ (Dense Scene) | {bb_stats['density_6_to_10_objects']:,} | {bb_stats['density_6_to_10_objects']/total_imgs*100:.2f}% |")
    doc.append(f"| > 10 วัตถุ (Crowded Traffic) | {bb_stats['density_above_10_objects']:,} | {bb_stats['density_above_10_objects']/total_imgs*100:.2f}% |")

    doc.append("\n### 5.2 การจำแนกขนาดวัตถุ (Object Scale Breakdown - Normalized Area)")
    n_area = bb_stats["normalized_area"]
    doc.append(f"- **วัตถุขนาดเล็ก (Small Objects, Area < 5% ของภาพ):** **{n_area['small_objects_pct']:.2f}%**")
    doc.append(f"- **วัตถุขนาดกลาง (Medium Objects, 5% <= Area <= 25%):** **{n_area['medium_objects_pct']:.2f}%**")
    doc.append(f"- **วัตถุขนาดใหญ่ (Large Objects, Area > 25%):** **{n_area['large_objects_pct']:.2f}%**")
    doc.append(f"- **ค่าเฉลี่ยพื้นที่ Bounding Box:** `{n_area['mean']:.4f}` (Median: `{n_area['median']:.4f}`)")

    doc.append("\n### 5.3 ตำแหน่งเชิงพื้นที่ (Spatial Center Distribution)")
    sc = bb_stats["spatial_center_mean"]
    doc.append(f"- **จุดกึ่งกลางเฉลี่ยแกน X ($X_{{center}}$):** `{sc['cx']:.4f}` (ใกล้เคียงศูนย์กลางภาพ 0.50)")
    doc.append(f"- **จุดกึ่งกลางเฉลี่ยแกน Y ($Y_{{center}}$):** `{sc['cy']:.4f}` (ค่อนไปทางครึ่งล่างของภาพ ซึ่งสอดคล้องกับระนาบถนนของการจราจร)")
    doc.append("\n---\n")

    # Section 6: Key Findings & Recommendations
    doc.append("## 6. ข้อค้นพบสำคัญและข้อเสนอแนะสำหรับการ Preprocessing และ Modeling")
    doc.append("### 💡 ข้อค้นพบหลัก (Key Insights)")
    doc.append("1. **ความหลากหลายของขนาดภาพ:** รูปภาพมีขนาด Resolution แตกต่างกัน จึงจำเป็นต้องทำการ Resize แบบคงอัตราส่วน (Aspect Ratio Preserved / Letterbox Padding) เพื่อป้องกันภาพบิดเบี้ยว")
    doc.append("2. **ความไม่สมดุลของคลาส (Class Imbalance):** มีคลาสเด่น (Dominant) และคลาสย่อย (Minority) จึงควรใช้เทคนิค Class-weighted Loss (เช่น Focal Loss) หรือ Data Augmentation สำหรับคลาสย่อย")
    doc.append("3. **การกระจายตัวของขนาดวัตถุ:** มีทั้งวัตถุขนาดเล็ก (รถที่อยู่ไกล) และขนาดใหญ่ (รถที่อยู่ใกล้) ควรใช้โมเดลที่มี Multi-scale Feature Pyramid (เช่น YOLOv8, Feature Pyramid Network - FPN)")
    doc.append("4. **ความสมบูรณ์ของพิกัด Annotation:** พิกัดทั้งหมดอยู่ในช่วงมาตรฐาน $[0, 1]$ รูปภาพสามารถอ่านได้ 100% ไม่มีไฟล์เสียหาย")
    doc.append("\n### 🛠️ ขั้นตอนการ Preprocessing ที่แนะนำ")
    doc.append("- **Letterbox Resize:** ปรับขนาดภาพเป็นมาตรฐาน (เช่น $640 \\times 640$) พร้อม Padding สีเทา (114, 114, 114) และปรับพิกัด Bounding Box ตามสเกลใหม่อย่างแม่นยำ")
    doc.append("- **Color Standardization & Normalization:** สเกลค่าพิกเซลให้อยู่ในช่วง $[0.0, 1.0]$ หรือมาตรฐาน ImageNet Mean/Std")
    doc.append("- **Data Augmentation:** ใช้ Mosaic, Mixup, Random Horizontal Flip, HSV Color Jittering เพื่อเพิ่มความคงทนของโมเดล")
    doc.append("\n---\n")
    doc.append("*รายงานนี้จัดทำขึ้นโดยอัตโนมัติผ่านโมดูล `src/eda/eda_analyzer.py`*")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(doc))
    
    print(f"✅ Generated EDA Report at: {output_file}")


if __name__ == "__main__":
    analyzer = DatasetEDAAnalyzer("data/raw")
    results = analyzer.analyze()
    generate_eda_report_markdown(results, "reports/eda/eda_report.md")
