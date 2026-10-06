import os
import cv2
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =====================================================================
# 1. GÖRÜNTÜ ÖN İŞLEME SİSTEMİ (BGR / TERMAL DOSTU)
# =====================================================================
class ImagePreprocessor:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def to_rgb_for_dino(self, bgr_crop):
        if bgr_crop is None or bgr_crop.size == 0:
            return None
        return cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)

    def to_thermal_for_dino(self, bgr_crop, mode="clahe"):
        if bgr_crop is None or bgr_crop.size == 0:
            return None
        gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)

        if mode == "clahe":
            gray = self.clahe.apply(gray)
        elif mode == "edges":
            gray = cv2.Canny(gray, 30, 100)
        elif mode == "invert":
            gray = 255 - gray

        rgb_thermal = cv2.merge([gray, gray, gray])
        return rgb_thermal


# =====================================================================
# 2. DINOv2 BATCH EMBEDDING EXTRACTOR
# =====================================================================
class FeatureExtractor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"📦 [DINOv2 AI Motoru] Cihaz: {self.device} üzerinde aktif.")

        yerel_repo = os.path.join(BASE_DIR, "models", "dinov2_repo")
        yerel_agirlik = os.path.join(BASE_DIR, "models", "dinov2_vits14_pretrain.pth")

        if not os.path.exists(yerel_repo) or not os.path.exists(yerel_agirlik):
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.device)
        else:
            self.model = torch.hub.load(
                yerel_repo,
                'dinov2_vits14',
                source='local',
                weights=yerel_agirlik
            ).to(self.device)

        if self.device.type in ["cuda", "mps"]:
            self.model = self.model.half()
        self.model.eval()
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_batch(self, list_of_rgb_crops, batch_size=64):
        if not list_of_rgb_crops:
            return np.empty((0, 384))
        all_embeddings = []
        for start in range(0, len(list_of_rgb_crops), batch_size):
            group = list_of_rgb_crops[start:start + batch_size]
            tensor_list = [self.transform(Image.fromarray(crop)) for crop in group]
            batch_tensor = torch.stack(tensor_list).to(self.device)
            if self.device.type in ["cuda", "mps"]:
                batch_tensor = batch_tensor.half()
            with torch.inference_mode():
                embeddings = self.model(batch_tensor)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                all_embeddings.append(embeddings.cpu().float().numpy())
        return np.concatenate(all_embeddings, axis=0)


# =====================================================================
# 3. ADAPTİF REFERANS ENCODER (RGB / TERMAL AYRIMLI & DINAMIK GEOMETRI)
# =====================================================================
class ReferenceEncoder:
    def __init__(self, feature_extractor):
        self.extractor = feature_extractor
        self.augmented_embeddings = []
        self.is_thermal = False
        self.ref_aspect_ratio = 1.0
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def detect_mode(self, ref_bgr):
        if ref_bgr is None or ref_bgr.size == 0:
            return False
        b, g, r = cv2.split(ref_bgr)
        diff_bg = np.mean(np.abs(b.astype(float) - g.astype(float)))
        diff_gr = np.mean(np.abs(g.astype(float) - r.astype(float)))

        if (diff_bg + diff_gr) < 4.0:
            self.is_thermal = True
            print("🌡️ [OTOMATİK MOD TESPİTİ] Termal / Grayscale mod aktifleşti.")
        else:
            self.is_thermal = False
            print("🌈 [OTOMATİK MOD TESPİTİ] Renkli (RGB) mod aktifleşti.")
        return self.is_thermal

    def set_reference(self, ref_bgr):
        if ref_bgr is None or ref_bgr.size == 0:
            raise ValueError("HATA: Referans görseli boş veya yüklenemedi!")

        h, w = ref_bgr.shape[:2]
        self.ref_aspect_ratio = float(w) / float(max(h, 1))
        print(f"📐 [OTOMATİK GEOMETRİ] Referans En-Boy Oranı: {self.ref_aspect_ratio:.2f}")

        self.detect_mode(ref_bgr)
        raw_variations = [ref_bgr.copy()]

        for angle in [-15, -5, 5, 15, 90, 180, 270]:
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            raw_variations.append(cv2.warpAffine(ref_bgr, M, (w, h), borderMode=cv2.BORDER_REPLICATE))

        raw_variations.append(cv2.flip(ref_bgr, 1))

        for scale in [0.3, 0.5, 0.75, 1.0, 1.3, 1.6, 2.2]:
            nh, nw = int(h * scale), int(w * scale)
            if nh < 10 or nw < 10:
                continue
            raw_variations.append(cv2.resize(ref_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR))

        pts1 = np.float32([[0,0], [w,0], [0,h], [w,h]])
        pts2 = np.float32([[int(w*0.05), int(h*0.05)], [int(w*0.95), int(h*0.02)], [int(w*0.02), int(h*0.98)], [int(w*0.98), int(h*0.95)]])
        M_persp = cv2.getPerspectiveTransform(pts1, pts2)
        raw_variations.append(cv2.warpPerspective(ref_bgr, M_persp, (w, h), borderMode=cv2.BORDER_REPLICATE))

        crops_rgb = []

        if self.is_thermal:
            for var in raw_variations:
                if var is None or var.size == 0:
                    continue
                gray = cv2.cvtColor(var, cv2.COLOR_BGR2GRAY)
                crops_rgb.append(cv2.merge([gray, gray, gray]))

                clahe_img = self.clahe.apply(gray)
                crops_rgb.append(cv2.merge([clahe_img, clahe_img, clahe_img]))

                edges = cv2.Canny(gray, 30, 120)
                crops_rgb.append(cv2.merge([edges, edges, edges]))

                invert = 255 - gray
                crops_rgb.append(cv2.merge([invert, invert, invert]))
            print(f"🎯 Termal Referans Bankası ({len(crops_rgb)} varyasyon) hafızaya alındı.")
        else:
            for alpha in [0.8, 1.25]:
                for beta in [-15, 15]:
                    raw_variations.append(cv2.convertScaleAbs(ref_bgr, alpha=alpha, beta=beta))

            crops_rgb = [cv2.cvtColor(var, cv2.COLOR_BGR2RGB) for var in raw_variations if var is not None and var.size > 0]
            print(f"🎯 Saf RGB Referans Bankası ({len(crops_rgb)} varyasyon) hafızaya alındı.")

        emb_rgb = self.extractor.extract_batch(crops_rgb)
        self.augmented_embeddings = emb_rgb / np.linalg.norm(emb_rgb, axis=-1, keepdims=True)


# =====================================================================
# 4. GENİŞ VE KOŞULLU ADAY ÜRETİCİ
# =====================================================================
class YoloProposalGenerator:
    def __init__(self, min_area_ratio=0.00015, max_area_ratio=0.85):
        self.model = YOLO(os.path.join(BASE_DIR, "models" if os.path.exists(os.path.join(BASE_DIR, "models")) else "", "yolo11n.pt"))
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio

    def generate_proposals(self, frame_bgr, roi=None):
        if frame_bgr is None:
            return []
        fh, fw, _ = frame_bgr.shape
        total_area = fh * fw
        search_area = frame_bgr if roi is None else frame_bgr[roi[1]:roi[3], roi[0]:roi[2]]
        offset_x = 0 if roi is None else roi[0]
        offset_y = 0 if roi is None else roi[1]

        results = self.model(search_area, conf=0.05, verbose=False)[0]
        raw_boxes, scores = [], []

        for box in results.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(xyxy[0]) + offset_x, int(xyxy[1]) + offset_y, int(xyxy[2]) + offset_x, int(xyxy[3]) + offset_y
            x1, y1 = max(0, min(x1, fw)), max(0, min(y1, fh))
            x2, y2 = max(0, min(x2, fw)), max(0, min(y2, fh))
            box_area = (x2 - x1) * (y2 - y1)

            if (box_area / total_area < self.min_area_ratio) or (box_area / total_area > self.max_area_ratio):
                continue
            raw_boxes.append([x1, y1, x2, y2])
            scores.append(float(box.conf[0]))

        gray = cv2.cvtColor(search_area, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        hsv = cv2.cvtColor(search_area, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 30, 30])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        combined_edges = cv2.bitwise_or(edges, green_mask)
        combined_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

        contours, _ = cv2.findContours(combined_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 15 or h < 15:
                continue
            x1, y1 = x + offset_x, y + offset_y
            x2, y2 = x1 + w, y1 + h
            box_area = w * h
            area_ratio = box_area / total_area
            aspect_ratio = w / max(h, 1)
            if not (self.min_area_ratio <= area_ratio <= self.max_area_ratio):
                continue
            if not (0.10 <= aspect_ratio <= 10.0):
                continue
            contour_boxes.append((box_area, [x1, y1, x2, y2]))

        for _, candidate in sorted(contour_boxes, reverse=True)[:35]:
            raw_boxes.append(candidate)
            scores.append(0.15)

        if not raw_boxes:
            return []
        nms_boxes = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in raw_boxes]
        indices = cv2.dnn.NMSBoxes(nms_boxes, scores, score_threshold=0.05, nms_threshold=0.45)
        return [raw_boxes[i] for i in indices.flatten()] if len(indices) > 0 else []

    def generate_selective_search(self, frame_bgr, roi=None, limit=250):
        fh, fw = frame_bgr.shape[:2]
        if roi is None:
            search_area = frame_bgr
            offset_x = offset_y = 0
        else:
            search_area = frame_bgr[roi[1]:roi[3], roi[0]:roi[2]]
            offset_x, offset_y = roi[0], roi[1]
        sh, sw = search_area.shape[:2]
        if sh < 20 or sw < 20:
            return []

        target_w = min(480, sw)
        target_h = max(1, round(sh * target_w / sw))
        resized = cv2.resize(search_area, (target_w, target_h))

        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(resized)
        ss.switchToSelectiveSearchFast()
        rects = ss.process()

        scale_x, scale_y = sw / target_w, sh / target_h
        proposals = []
        seen = set()
        for x, y, w, h in rects:
            x1 = int(x * scale_x) + offset_x
            y1 = int(y * scale_y) + offset_y
            x2 = int((x + w) * scale_x) + offset_x
            y2 = int((y + h) * scale_y) + offset_y
            area_ratio = ((x2 - x1) * (y2 - y1)) / float(fw * fh)
            aspect = (x2 - x1) / max(y2 - y1, 1)
            box = (x1, y1, x2, y2)
            if not (self.min_area_ratio <= area_ratio <= self.max_area_ratio):
                continue
            if not (0.10 <= aspect <= 10.0) or box in seen:
                continue
            seen.add(box)
            proposals.append(list(box))
            if len(proposals) >= limit:
                break
        return proposals


# =====================================================================
# 5. SEMANTİK EŞLEŞTİRİCİ (OTOMATİK ŞEKİL & BOYUT KONTROLLÜ)
# =====================================================================
class SemanticMatcher:
    def __init__(self, min_acceptable_similarity=0.55, margin_threshold=0.04):
        self.min_acceptable_similarity = min_acceptable_similarity
        self.margin_threshold = margin_threshold

    def calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

    def evaluate_batch(self, ref_embs, crop_embs, proposals, frame_shape, ref_aspect_ratio=1.0, min_similarity=None, margin=None):
        if len(proposals) == 0 or crop_embs.shape[0] == 0:
            return None, 0.0, "Eşleşme Yok"

        similarity_matrix = np.dot(crop_embs, ref_embs.T)
        crop_scores = np.max(similarity_matrix, axis=1)

        fh, fw = frame_shape[:2]
        total_area = fh * fw
        adjusted_scores = []

        for idx, score in enumerate(crop_scores):
            box = proposals[idx]
            bw = box[2] - box[0]
            bh = box[3] - box[1]
            box_area = bw * bh
            area_ratio = box_area / total_area

            cand_aspect_ratio = float(bw) / float(max(bh, 1))
            ratio_difference = abs(np.log(cand_aspect_ratio / ref_aspect_ratio))
            aspect_penalty = 0.15 * ratio_difference

            if ref_aspect_ratio > 1.5 or ref_aspect_ratio < 0.65:
                area_penalty = 0.12 * area_ratio
            else:
                area_penalty = 0.02 * area_ratio

            adjusted_scores.append(score - aspect_penalty - area_penalty)

        adjusted_scores = np.array(adjusted_scores)
        sorted_indices = np.argsort(adjusted_scores)[::-1]

        best_idx = sorted_indices[0]
        best_score = crop_scores[best_idx]

        min_similarity = self.min_acceptable_similarity if min_similarity is None else min_similarity
        margin = self.margin_threshold if margin is None else margin

        if best_score < min_similarity:
            return None, best_score, "Eşleşme Yok"

        if len(sorted_indices) > 1:
            second_idx = sorted_indices[1]
            second_score = crop_scores[second_idx]
            iou = self.calculate_iou(proposals[best_idx], proposals[second_idx])

            if iou < 0.45:
                if (best_score - second_score) < margin:
                    return None, best_score, "Kararsız Sahne (Reddedildi)"

        return proposals[best_idx], best_score, "Yüksek Güven"


# =====================================================================
# 6. CSRT GÖRSEL TAKİP + GÜVENLİ KALMAN FILTER
# =====================================================================
class CSRTReferenceTracker:
    def __init__(self):
        self.tracker = None
        self.is_active = False

    def start(self, frame_bgr, box):
        x1, y1, x2, y2 = [int(v) for v in box]
        if x2 <= x1 or y2 <= y1:
            return
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame_bgr, (x1, y1, x2 - x1, y2 - y1))
        self.is_active = True

    def update(self, frame_bgr):
        if not self.is_active or self.tracker is None:
            return None
        success, tracked = self.tracker.update(frame_bgr)
        if not success:
            self.reset()
            return None
        x, y, w, h = [int(round(v)) for v in tracked]
        return [x, y, x + w, y + h]

    def reset(self):
        self.tracker = None
        self.is_active = False


class KalmanVisualTracker:
    def __init__(self, min_hits=3, max_lost=6, min_similarity_threshold=0.55):
        self.min_hits = min_hits
        self.max_lost = max_lost
        self.min_similarity_threshold = min_similarity_threshold
        self.hit_streak, self.lost_count = 0, 0
        self.is_active = False
        self.is_confirmed = False
        self.last_w, self.last_h = 100, 100
        self.last_detected_box = None
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

    def update(self, detected_box, similarity_score):
        if detected_box is None or similarity_score < self.min_similarity_threshold:
            self.lost_count += 1
            self.hit_streak = max(0, self.hit_streak - 1)
            prediction = self.kalman.predict()

            if self.is_active and self.lost_count <= self.max_lost:
                px, py = int(prediction[0][0]), int(prediction[1][0])
                return [px - self.last_w//2, py - self.last_h//2, px + self.last_w//2, py + self.last_h//2], False

            self.is_active = False
            self.is_confirmed = False
            return None, False

        x1, y1, x2, y2 = detected_box
        cx, cy = float((x1 + x2) / 2), float((y1 + y2) / 2)
        self.last_w, self.last_h = (x2 - x1), (y2 - y1)
        self.last_detected_box = detected_box

        if not self.is_active:
            self.kalman.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
            self.hit_streak, self.lost_count = 1, 0
            self.is_active = True
            self.is_confirmed = False
        else:
            weight = np.clip(similarity_score, 0.1, 1.0)
            self.kalman.measurementNoiseCov = (np.eye(2, dtype=np.float32) * (0.1 / weight)).astype(np.float32)
            self.kalman.correct(np.array([[cx], [cy]], np.float32))
            self.kalman.predict()
            self.hit_streak += 1
            self.lost_count = 0

        if self.hit_streak >= self.min_hits:
            self.is_confirmed = True
        return detected_box, self.is_confirmed


# =====================================================================
# YARIŞMA İÇİN ÇATI (WRAPPER) SINIFI
# =====================================================================
class ReferenceManager:
    """
    Yarışma sunucusu her kareyi tek tek yollar. AI modellerini her karede
    baştan yüklememek için bu sınıf modelleri RAM'de canlı tutar.
    """
    def __init__(self):
        print("🚀 Referans Arama Motoru Başlatılıyor...")
        self.preprocessor = ImagePreprocessor()
        self.extractor = FeatureExtractor()
        self.proposal_gen = YoloProposalGenerator()
        self.matcher = SemanticMatcher()

        # Hafıza (Cache & States)
        self.ref_embeddings_cache = {}
        self.ref_aspect_ratios = {}
        self.is_thermal_modes = {}
        self.csrt_trackers = {}
        self.kalman_trackers = {}
        self.last_valid_boxes = {}

    def find_reference(self, frame, ref_img, ref_url, is_thermal=False):
        """Tek bir karede referansı arar ve bbox döndürür."""

        # 1. Referans Kodlama (Eğer önbellekte yoksa otomatik kodlama yapar)
        if ref_url not in self.ref_embeddings_cache:
            if ref_img is None:
                return None
            encoder = ReferenceEncoder(self.extractor)
            encoder.set_reference(ref_img)
            self.ref_embeddings_cache[ref_url] = encoder.augmented_embeddings
            self.ref_aspect_ratios[ref_url] = encoder.ref_aspect_ratio
            self.is_thermal_modes[ref_url] = encoder.is_thermal
            print(f"🎯 Yeni Referans Belleğe Alındı: {ref_url.split('/')[-1]}")

        ref_embeddings = self.ref_embeddings_cache[ref_url]
        ref_aspect_ratio = self.ref_aspect_ratios[ref_url]
        is_thermal_mode = self.is_thermal_modes[ref_url]

        # 2. Referansa Özel Filtreleri Al
        if ref_url not in self.csrt_trackers:
            self.csrt_trackers[ref_url] = CSRTReferenceTracker()
            min_similarity_threshold = 0.45 if is_thermal_mode else 0.55
            self.kalman_trackers[ref_url] = KalmanVisualTracker(min_similarity_threshold=min_similarity_threshold)
            self.last_valid_boxes[ref_url] = None

        csrt_tracker = self.csrt_trackers[ref_url]
        tracker = self.kalman_trackers[ref_url]
        last_valid_box = self.last_valid_boxes[ref_url]

        # Mod bazlı dinamik eşik konfigürasyonu
        if is_thermal_mode:
            STRICT_CSRT_THRESHOLD = 0.50
            STRICT_YOLO_THRESHOLD = 0.55
            STRICT_SS_THRESHOLD = 0.60
            MIN_ACCEPTABLE_SIMILARITY = 0.45
        else:
            STRICT_CSRT_THRESHOLD = 0.60
            STRICT_YOLO_THRESHOLD = 0.65
            STRICT_SS_THRESHOLD = 0.70
            MIN_ACCEPTABLE_SIMILARITY = 0.49

        # 3. Arama Alanı (ROI) Belirleme
        roi = None
        takip_kutusu = last_valid_box if last_valid_box is not None else tracker.last_detected_box
        if takip_kutusu is not None:
            fh, fw, _ = frame.shape
            # 🟢 KESİN DÜZELTME: int tip dönüşümleri listenin içindeki elemanlara ayrı ayrı uygulanmıştır
            roi = [
                max(0, int(takip_kutusu[0]) - 150),
                max(0, int(takip_kutusu[1]) - 150),
                min(fw, int(takip_kutusu[2]) + 150),
                min(fh, int(takip_kutusu[3]) + 150)
            ]

        best_box, best_score, conf_level = None, 0.0, "Eşleşme Yok"
        taze_dogrulama = False

        # 4. CSRT Takibi
        tracked_box = csrt_tracker.update(frame)
        if tracked_box is not None:
            fh, fw = frame.shape[:2]
            tx1 = max(0, min(tracked_box[0], fw - 1))
            ty1 = max(0, min(tracked_box[1], fh - 1))
            tx2 = max(0, min(tracked_box[2], fw))
            ty2 = max(0, min(tracked_box[3], fh))

            bgr_crop = frame[ty1:ty2, tx1:tx2]
            if bgr_crop.size > 0:
                if is_thermal_mode:
                    rgb_crop = self.preprocessor.to_thermal_for_dino(bgr_crop, mode="clahe")
                else:
                    rgb_crop = self.preprocessor.to_rgb_for_dino(bgr_crop)

                tracked_embedding = self.extractor.extract_batch([rgb_crop])
                tracked_score = float(np.max(np.dot(tracked_embedding, ref_embeddings.T)))

                if tracked_score >= STRICT_CSRT_THRESHOLD:
                    best_box = [tx1, ty1, tx2, ty2]
                    best_score = tracked_score
                    conf_level = "CSRT + DINO"
                    taze_dogrulama = True
                else:
                    csrt_tracker.reset()

        # 5. YOLO & Kontur Hattı
        if best_box is None:
            proposals = self.proposal_gen.generate_proposals(frame, roi=roi)
            if roi is not None and len(proposals) == 0:
                proposals = self.proposal_gen.generate_proposals(frame, roi=None)

            if len(proposals) > 0:
                if is_thermal_mode:
                    rgb_crops = [self.preprocessor.to_thermal_for_dino(frame[p[1]:p[3], p[0]:p[2]], mode="clahe") for p in proposals]
                else:
                    rgb_crops = [self.preprocessor.to_rgb_for_dino(frame[p[1]:p[3], p[0]:p[2]]) for p in proposals]

                crop_embeddings = self.extractor.extract_batch(rgb_crops)
                best_box, best_score, conf_level = self.matcher.evaluate_batch(
                    ref_embeddings, crop_embeddings, proposals, frame.shape,
                    ref_aspect_ratio=ref_aspect_ratio, min_similarity=STRICT_YOLO_THRESHOLD
                )
                if best_box is not None:
                    taze_dogrulama = True

        # 6. Koşullu Fallback: Selective Search
        absurd_box = False
        if best_box is not None:
            fh, fw = frame.shape[:2]
            bw, bh_box = (best_box[2] - best_box[0]), (best_box[3] - best_box[1])
            if (bw < 15 or bh_box < 15) or (bw > fw * 0.85 or bh_box > fh * 0.85):
                absurd_box = True

        if best_box is None or absurd_box:
            selective_proposals = self.proposal_gen.generate_selective_search(frame, roi=roi)
            if len(selective_proposals) > 0:
                if is_thermal_mode:
                    rgb_selective_crops = [self.preprocessor.to_thermal_for_dino(frame[p[1]:p[3], p[0]:p[2]], mode="clahe") for p in selective_proposals]
                else:
                    rgb_selective_crops = [self.preprocessor.to_rgb_for_dino(frame[p[1]:p[3], p[0]:p[2]]) for p in selective_proposals]

                selective_embeddings = self.extractor.extract_batch(rgb_selective_crops)
                sel_box, sel_score, sel_conf = self.matcher.evaluate_batch(
                    ref_embeddings,
                    selective_embeddings,
                    selective_proposals,
                    frame.shape,
                    ref_aspect_ratio=ref_aspect_ratio,
                    min_similarity=STRICT_SS_THRESHOLD,
                    margin=0.0
                )
                if sel_box is not None:
                    best_box = sel_box
                    best_score = sel_score
                    conf_level = "Selective Search + DINO"
                    taze_dogrulama = True

        # Nihai Güvenlik Sınırı Kontrolü
        if best_score < MIN_ACCEPTABLE_SIMILARITY:
            best_box = None
            best_score = 0.0
            conf_level = "Eşleşme Yok"
            taze_dogrulama = False
            csrt_tracker.reset()

        if best_box is not None and not csrt_tracker.is_active:
            csrt_tracker.start(frame, best_box)

        # Kalman Takibi Güncellemesi
        kalman_box, is_validated = tracker.update(best_box, best_score)

        # Sonuç Dönüşü
        if taze_dogrulama and best_box is not None:
            fh, fw = frame.shape[:2]
            tl_x = float(max(0, min(int(best_box[0]), fw - 1)))
            tl_y = float(max(0, min(int(best_box[1]), fh - 1)))
            br_x = float(max(0, min(int(best_box[2]), fw - 1)))
            br_y = float(max(0, min(int(best_box[3]), fh - 1)))

            if br_x <= tl_x or br_y <= tl_y:
                self.last_valid_boxes[ref_url] = None
                return None

            validated_box = [tl_x, tl_y, br_x, br_y]
            self.last_valid_boxes[ref_url] = validated_box
            return validated_box
        else:
            self.last_valid_boxes[ref_url] = None
            return None