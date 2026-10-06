import cv2
import numpy as np
import os
from ultralytics import YOLO

# =====================================================================
# YARDIMCI FONKSİYONLAR
# =====================================================================
def check_overlap(boxA, boxB):
    """İki kutunun kesişip kesişmediğini kontrol eder."""
    ix1 = max(boxA[0], boxB[0])
    iy1 = max(boxA[1], boxB[1])
    ix2 = min(boxA[2], boxB[2])
    iy2 = min(boxA[3], boxB[3])
    if ix1 < ix2 and iy1 < iy2:
        return True
    return False

def has_foreign_object(crop_img):
    """
    YOLO'NUN KAÇIRDIĞI İNSAN/ARAÇLAR İÇİN GÜVENLİK DUVARI:
    Alanın %1'inden fazlası koyu/karanlık ise "Yabancı Cisim" sayar.
    """
    if crop_img.shape[0] < 10 or crop_img.shape[1] < 10:
        return False

    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
    lower_dark = np.array([0, 20, 0])
    upper_dark = np.array([180, 255, 80])
    dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)

    dark_ratio = cv2.countNonZero(dark_mask) / float(crop_img.shape[0] * crop_img.shape[1])

    if dark_ratio > 0.01:
        return True
    return False

def check_text_presence(crop_img):
    """
    Sadece UAP/UAİ'nin merkezinde harflerin oluşturduğu keskin kontrast
    çizgileri (kenar yoğunluğu) var mı diye bakar.
    """
    if crop_img.shape[0] < 15 or crop_img.shape[1] < 15:
        return False

    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    margin_y, margin_x = int(h * 0.20), int(w * 0.20)
    center_roi = gray[margin_y:h-margin_y, margin_x:w-margin_x]

    if center_roi.size == 0:
        return False

    blurred = cv2.GaussianBlur(center_roi, (3, 3), 0)
    edges = cv2.Canny(blurred, 60, 150)
    edge_density = np.count_nonzero(edges) / float(center_roi.size)

    if edge_density > 0.015:
        return True
    return False

# =====================================================================
# GÖREV 1 ANA MOTORU (YARIŞMA İÇİN SINIF YAPISI)
# =====================================================================
class DetectionManager:
    def __init__(self):
        print("🚀 Görev 1 Motoru (YOLO & Optik Akış) Başlatılıyor...")

        # Model yollarını projenin yapısına göre belirt
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Eğer modeller src klasöründeyse:
        model_uap_path = os.path.join(base_dir,"models", "uai_uap_rgb.pt")
        model_obs_path = os.path.join(base_dir,"models", "rgb_main.pt")

        self.model_uap = YOLO(model_uap_path)
        self.model_obs = YOLO(model_obs_path)

        self.motion_threshold = 3.0
        self.edge_margin = 30
        self.uap_edge_margin = 2
        self.lock_consecutive_frames = 5

        self.old_gray = None
        self.p0 = None
        self.vehicle_history = {}
        self.track_states = {}

        self.feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=7)
        self.lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def process_frame(self, frame):
        """
        Sunucudan gelen tek bir kareyi alır, tespitleri ve hareket durumlarını
        hesaplar, yarışma sözlüğüne uygun anahtarlarla geri döndürür.
        """
        results = [] # Sunucuya gönderilecek formatlanmış objeler buraya eklenecek

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_height, frame_width = frame.shape[:2]
        frame_clean = frame.copy()

        # -------------------------------------------------------------
        # 1. KAMERA HAREKETİ (EGO-MOTION) HESAPLAMA
        # -------------------------------------------------------------
        transform_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        if self.old_gray is not None:
            if self.p0 is None or len(self.p0) < 10:
                self.p0 = cv2.goodFeaturesToTrack(self.old_gray, mask=None, **self.feature_params)

            if self.p0 is not None:
                p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, curr_gray, self.p0, None, **self.lk_params)
                good_new = p1[st == 1]
                good_old = self.p0[st == 1]

                if len(good_new) >= 4:
                    M, _ = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC)
                    if M is not None:
                        transform_matrix = M

                self.p0 = good_new.reshape(-1, 1, 2)
        else:
            self.p0 = cv2.goodFeaturesToTrack(curr_gray, mask=None, **self.feature_params)

        self.old_gray = curr_gray.copy()

        # -------------------------------------------------------------
        # 2. ARAÇ VE İNSAN TESPİTİ + TAKİP
        # -------------------------------------------------------------
        obs_results = self.model_obs.track(frame_clean, persist=True, verbose=False)
        obstacles = []

        if obs_results[0].boxes.id is not None:
            boxes = obs_results[0].boxes.xyxy.cpu().numpy()
            track_ids = obs_results[0].boxes.id.cpu().numpy()
            classes = obs_results[0].boxes.cls.cpu().numpy()

            for box, track_id, cls_id in zip(boxes, track_ids, classes):
                x1, y1, x2, y2 = map(int, box)
                obstacles.append((x1, y1, x2, y2))

                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                current_center = np.array([cx, cy, 1.0])

                if int(cls_id) == 0: # 0: Tasit
                    cls_key = "Tasit"
                    landing_key = "Inis Alani Degil"

                    is_on_edge = (x1 <= self.edge_margin or y1 <= self.edge_margin or
                                  x2 >= frame_width - self.edge_margin or y2 >= frame_height - self.edge_margin)

                    if track_id not in self.track_states:
                        self.track_states[track_id] = {
                            "locked": False,
                            "final_status": "Sabit",
                            "moving_streak": 0,
                            "stationary_streak": 0,
                            "current_status": "Sabit"
                        }

                    state = self.track_states[track_id]

                    if state["locked"]:
                        moving_key = state["final_status"]
                    else:
                        if track_id in self.vehicle_history:
                            prev_cx, prev_cy = self.vehicle_history[track_id]
                            prev_center = np.array([prev_cx, prev_cy, 1.0])
                            expected_center = transform_matrix @ prev_center
                            dist = np.linalg.norm(current_center[:2] - expected_center)

                            if not is_on_edge:
                                if dist > self.motion_threshold:
                                    state["moving_streak"] += 1
                                    state["stationary_streak"] = 0
                                    state["current_status"] = "Hareketli"

                                    if state["moving_streak"] >= self.lock_consecutive_frames:
                                        state["locked"] = True
                                        state["final_status"] = "Hareketli"
                                else:
                                    state["stationary_streak"] += 1
                                    state["moving_streak"] = 0
                                    state["current_status"] = "Sabit"

                                    if state["stationary_streak"] >= self.lock_consecutive_frames:
                                        state["locked"] = True
                                        state["final_status"] = "Sabit"

                        moving_key = state["current_status"]

                    self.vehicle_history[track_id] = (cx, cy)

                    results.append({
                        "cls_key": cls_key,
                        "landing_key": landing_key,
                        "moving_key": moving_key,
                        "bbox": [x1, y1, x2, y2]
                    })

                else: # 1: Insan
                    cls_key = "Insan"
                    landing_key = "Inis Alani Degil"
                    moving_key = "Tasit Degil"

                    results.append({
                        "cls_key": cls_key,
                        "landing_key": landing_key,
                        "moving_key": moving_key,
                        "bbox": [x1, y1, x2, y2]
                    })

        # -------------------------------------------------------------
        # 3. UAP / UAİ TESPİTİ VE İNİŞ DURUMU
        # -------------------------------------------------------------
        uap_results = self.model_uap.predict(frame_clean, verbose=False)

        if len(uap_results[0].boxes) > 0:
            boxes = uap_results[0].boxes.xyxy.cpu().numpy()
            classes = uap_results[0].boxes.cls.cpu().numpy()

            for box, cls_id in zip(boxes, classes):
                x1, y1, x2, y2 = map(int, box)

                # Model Sınıf İsimlerinin Garantili Eşleşmesi
                model_lbl = self.model_uap.names[int(cls_id)].upper()
                if "UAP" in model_lbl or int(cls_id) == 2:
                    cls_key = "UAP"
                else:
                    cls_key = "UAI"

                moving_key = "Tasit Degil"
                landing_key = "Inilebilir"

                # Aşama 1: Kenar Sınırı (2 Piksel)
                if (x1 <= self.uap_edge_margin or y1 <= self.uap_edge_margin or
                    x2 >= frame_width - self.uap_edge_margin or y2 >= frame_height - self.uap_edge_margin):
                    landing_key = "Inilemez"

                # Aşama 2: Kesişim
                if landing_key == "Inilebilir":
                    for obs_box in obstacles:
                        if check_overlap((x1, y1, x2, y2), obs_box):
                            landing_key = "Inilemez"
                            break

                # Aşama 2.5: Gölgeler ve Yabancı Cisimler
                if landing_key == "Inilebilir":
                    crop_y1, crop_y2 = max(0, y1), min(frame_height, y2)
                    crop_x1, crop_x2 = max(0, x1), min(frame_width, x2)
                    crop_img = frame_clean[crop_y1:crop_y2, crop_x1:crop_x2]

                    if has_foreign_object(crop_img):
                        landing_key = "Inilemez"

                # Aşama 3: İç Dokudaki Harfler
                if landing_key == "Inilebilir":
                    if not check_text_presence(crop_img):
                        landing_key = "Inilemez"

                results.append({
                    "cls_key": cls_key,
                    "landing_key": landing_key,
                    "moving_key": moving_key,
                    "bbox": [x1, y1, x2, y2]
                })

        return results