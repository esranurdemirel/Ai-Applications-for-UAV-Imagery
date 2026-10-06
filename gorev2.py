import cv2
import numpy as np

# =====================================================================
# SENİN ALGORİTMAN: FEATURE / OPTICAL FLOW YÖNETİMİ
# =====================================================================
class FeatureTracker:
    def __init__(self, feature_params=None, lk_params=None, min_features=60):
        self.feature_params = feature_params or dict(
            maxCorners=300, qualityLevel=0.1, minDistance=10, blockSize=7
        )
        self.lk_params = lk_params or dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        self.min_features = min_features

    def detect(self, gray_frame):
        return cv2.goodFeaturesToTrack(gray_frame, mask=None, **self.feature_params)

    def track(self, old_gray, new_gray, p0):
        if p0 is None or len(p0) == 0:
            p0 = self.detect(old_gray)
            if p0 is None:
                return None, None
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **self.lk_params)
        good_new = p1[st == 1] if p1 is not None else np.array([])
        good_old = p0[st == 1] if p1 is not None else np.array([])

        if p1 is None or len(good_new) < self.min_features:
            p0 = self.detect(old_gray)
            if p0 is None:
                return None, None
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **self.lk_params)
            if p1 is None:
                return None, None
            good_new = p1[st == 1]
            good_old = p0[st == 1]

        if len(good_new) < 4:
            return None, None
        return good_old, good_new


# =====================================================================
# SENİN ALGORİTMAN: ESSENTIAL MATRIX + ÖLÇÜM ÇIKARIMI
# =====================================================================
class MeasurementExtractor:
    def __init__(self):
        self.cum_yaw = 0.0

    @staticmethod
    def _build_feature_vector(dx, dy, dz):
        return np.array([dx, dy, dz, dx * dz, dy * dz, dz * dz, 1.0])

    def extract(self, good_old, good_new, K):
        """Essential Matrix ile delta ölçümü"""
        if good_old is None or len(good_old) < 8:
            return np.zeros(7)

        E, mask = cv2.findEssentialMat(good_old, good_new, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            return np.zeros(7)

        _, R, t, _ = cv2.recoverPose(E, good_old, good_new, K)

        tx, ty, tz = t.ravel()
        dx = tx * 80
        dy = ty * 80
        dz = tz * 40

        return self._build_feature_vector(dx, dy, dz)


# =====================================================================
# SENİN ALGORİTMAN: REGRESYON (Sadece Delta)
# =====================================================================
def fit_regression(X_list, Y_list):
    X = np.array(X_list)
    Y = np.array(Y_list)
    beta, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None)
    return beta

def compute_step_bias(X_list, Y_list, beta):
    X = np.array(X_list)
    Y = np.array(Y_list)
    predictions = X @ beta
    residuals = predictions - Y
    return np.mean(residuals, axis=0)


# =====================================================================
# SENİN ALGORİTMAN: EKF
# =====================================================================
class SimpleEKF:
    def __init__(self):
        self.x = np.zeros(3)
        self.P = np.eye(3) * 10
        self.Q = np.eye(3) * 0.01
        self.R = np.eye(3) * 0.2

    def update(self, z):
        self.P = self.P + self.Q
        K = self.P @ np.linalg.inv(self.P + self.R)
        self.x = self.x + K @ (z - self.x)
        self.P = (np.eye(3) - K) @ self.P
        return self.x.copy()


# =====================================================================
# YARIŞMA İÇİN KARE-KARE ÇAĞIRILACAK ANA SINIF
# =====================================================================
class VOEstimator:
    def __init__(self):
        self.tracker = FeatureTracker()
        self.extractor = MeasurementExtractor()
        self.ekf = SimpleEKF()

        # Kamera ve Görüntü Hafızası
        self.old_gray = None
        self.p0 = None
        self.K = None

        # Kalibrasyon ve Regresyon Verileri
        self.RAW_DELTAS = []
        self.GT_DELTAS = []
        self.beta_delta = None
        self.delta_bias = np.zeros(3)
        self.is_trained = False

        # Durum Değişkenleri
        self.est_pos_delta = np.array([0.0, 0.0, 0.0])
        self.prev_gt = None

    def process_frame(self, frame_path, gt_x, gt_y, gt_z, health_status):
        """Yarışma esnasında her bir kare (frame) için 1 kez çalışır."""
        frame = cv2.imread(frame_path)
        if frame is None:
            return self.est_pos_delta[0], self.est_pos_delta[1], self.est_pos_delta[2]

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Kamera Matrisi (K) başlatma
        if self.K is None:
            try:
                from camera import Camera
                self.K = Camera().camera_matrix(frame)
            except ImportError:
                # camera.py yoksa standart bir matris atar
                h, w = frame.shape[:2]
                self.K = np.array([[2792.2, 0, w/2], [0, 2795.2, h/2], [0, 0, 1]], dtype=np.float64)

        # -------------------------------------------------------------
        # 1. KARE VE BAŞLATMA DURUMU
        # -------------------------------------------------------------
        if self.old_gray is None:
            self.old_gray = frame_gray.copy()
            self.p0 = self.tracker.detect(self.old_gray)
            if health_status == '1' and gt_x is not None:
                self.est_pos_delta = np.array([float(gt_x), float(gt_y), float(gt_z)])
                self.prev_gt = self.est_pos_delta.copy()
            return self.est_pos_delta[0], self.est_pos_delta[1], self.est_pos_delta[2]

        # -------------------------------------------------------------
        # OPTİK AKIŞ VE VEKTÖR ÇIKARIMI (Senin Algoritman)
        # -------------------------------------------------------------
        good_old, good_new = self.tracker.track(self.old_gray, frame_gray, self.p0)
        delta_vec = self.extractor.extract(good_old, good_new, self.K) if good_old is not None else np.zeros(7)

        current_gt = None
        if health_status == '1' and gt_x is not None:
            current_gt = np.array([float(gt_x), float(gt_y), float(gt_z)])

        # -------------------------------------------------------------
        # AŞAMA 1: SAĞLIKLI GPS DURUMU (ÖĞRENME AŞAMASI)
        # -------------------------------------------------------------
        if health_status == '1':
            if current_gt is not None and self.prev_gt is not None:
                gt_delta = current_gt - self.prev_gt
                self.RAW_DELTAS.append(delta_vec)
                self.GT_DELTAS.append(gt_delta)

            if current_gt is not None:
                self.est_pos_delta = current_gt.copy()
                self.prev_gt = current_gt.copy()

            # Yeniden kalibre edilmesi gerektiğini işaretliyoruz (Çünkü veri güncellendi)
            self.is_trained = False

            p_final = self.est_pos_delta

        # -------------------------------------------------------------
        # AŞAMA 2: GPS KOPTU (KÖR UÇUŞ VE TAHMİN)
        # -------------------------------------------------------------
        elif health_status == '0':
            # Eğitim yapılmamışsa ve havuzda yeterince veri varsa eğit
            if not self.is_trained and len(self.RAW_DELTAS) >= 10:
                self.beta_delta = fit_regression(self.RAW_DELTAS, self.GT_DELTAS)
                self.delta_bias = compute_step_bias(self.RAW_DELTAS, self.GT_DELTAS, self.beta_delta)
                self.is_trained = True

            # Eğitilmiş beta matrisi ile Tahmin (Senin mantığın)
            if self.is_trained and self.beta_delta is not None:
                est_delta = (delta_vec @ self.beta_delta) - self.delta_bias
                self.est_pos_delta = self.est_pos_delta + est_delta

                # Sadece tahminde EKF güncellenir
                p_final = self.ekf.update(self.est_pos_delta)
            else:
                p_final = self.est_pos_delta

        else:
            p_final = self.est_pos_delta

        # -------------------------------------------------------------
        # SONRAKİ KARE HAZIRLIĞI
        # -------------------------------------------------------------
        self.old_gray = frame_gray.copy()
        if good_new is not None and len(good_new) > 0:
            self.p0 = good_new.reshape(-1, 1, 2)
        else:
            self.p0 = self.tracker.detect(frame_gray)

        return p_final[0], p_final[1], p_final[2]