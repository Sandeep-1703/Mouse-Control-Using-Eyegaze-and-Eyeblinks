# ULTIMATE DLIB Eye Mouse Control - 5 POINT CALIBRATION
# With SHORT BLINK = Single Click and LONG BLINK = Double Click (time-based)
# NEW: Mouth RE-ARMING band toggles ON/OFF; closing only re-arms.
# NEW: Live sensitivity controls that affect cursor speed/range from eye motion:
#   [ / ] : EDGE_GAMMA (lower -> more sensitive edges/corners)
#   ; / ' : SENSITIVITY_GAIN (higher -> larger movement)
#   - / = : MOUSE_HISTORY_SIZE (lower -> snappier)
#   0     : SNAP mouse move (duration=0) toggle

import cv2
import dlib
import pyautogui
import numpy as np
import time
import math

pyautogui.FAILSAFE = False

# ============================================================================
# ================== 🎛️ TUNABLE PARAMETERS - CUSTOMIZE HERE =================
# ============================================================================

# ============ SMOOTHING PARAMETERS ============
SMOOTH_FACTOR = 0.3
MOUSE_HISTORY_SIZE = 10

# ============ BLINK DETECTION (time-based) ============
EAR_THRESHOLD = 0.18
SHORT_MIN_FRAMES =  2   # minimal frames eyes must be closed to count
LONG_BLINK_MIN_SEC = 0.50# >= this => LONG blink
BLINK_REFRACTORY_SEC = 1 # cooldown between blink actions

# ============ IRIS DETECTION PARAMETERS ============
IRIS_THRESHOLD = 60
BILATERAL_D = 15
BILATERAL_SIGMA = 15

# ============ CAMERA SETTINGS ============
CAM_WIDTH = 640
CAM_HEIGHT = 480
MIRROR_FRAME = True

# ============ CALIBRATION PARAMETERS ============
SAMPLES_PER_CALIBRATION_POINT = 50
MIN_SAMPLES_FOR_VALID_CALIBRATION = 20
MIN_CALIBRATION_POINTS_FOR_ACCURACY = 5

# ============ MOUTH TOGGLE PARAMETERS ============
# Toggle happens on CLOSED -> (>= CLOSE_THR) rising edge (entering RE-ARM/OPEN), with cooldown.
MOUTH_OPEN_THR = 0.60
MOUTH_CLOSE_THR = 0.50
MOUTH_TOGGLE_COOLDOWN = 0.80
MOUTH_HYSTERESIS = 0.02

# ============ GAZE→CURSOR SENSITIVITY (NEW) ============
# These directly affect how far/fast the cursor moves for a given eye motion.
EDGE_GAMMA = 0.75         # <1 expands edges (more reach near corners); >1 compresses
SENSITIVITY_GAIN = 1.00   # global scale; >1 moves further, <1 moves less
SNAP_MOVE = False         # if True, pyautogui.moveTo(..., duration=0)

# ============================================================================

class UltimateDlibEyeMouse:
    def __init__(self):
        # Initialize DLIB
        self.detector = dlib.get_frontal_face_detector()
        try:
            self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        except Exception:
            print("❌ Error: Cannot find 'shape_predictor_68_face_landmarks.dat'")
            print("Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            raise
        
        # Screen dimensions
        self.screen_w, self.screen_h = pyautogui.size()
        
        # DLIB landmark indices
        self.LEFT_EYE_POINTS = list(range(36, 42))
        self.RIGHT_EYE_POINTS = list(range(42, 48))
        self.MOUTH_INNER_POINTS = list(range(60, 68))

        # Calibration
        self.is_calibrated = False
        self.cal_left_iris = {}
        self.cal_right_iris = {}
        self.cal_screen_points = {}
        self.calibration_count = 0
        
        self.CALIBRATION_POINTS = {
            1: (0.5, 0.5, "CENTER"),
            2: (0.1, 0.1, "TOP-LEFT"),
            3: (0.9, 0.1, "TOP-RIGHT"),
            4: (0.1, 0.9, "BOTTOM-LEFT"),
            5: (0.9, 0.9, "BOTTOM-RIGHT"),
        }
        
        # Mouse control
        self.mouse_active = False
        self.last_mouse_x = self.screen_w // 2
        self.last_mouse_y = self.screen_h // 2
        
        # Smoothing
        self.smooth_factor = SMOOTH_FACTOR
        self.mouse_history = []
        self.history_size = MOUSE_HISTORY_SIZE
        
        # Blink detection (time based)
        self.ear_threshold = EAR_THRESHOLD
        self.short_min_frames = SHORT_MIN_FRAMES
        self.long_min_sec = LONG_BLINK_MIN_SEC
        self.blink_refractory_sec = BLINK_REFRACTORY_SEC
        self.blink_started = False
        self.blink_start_time = 0.0
        self.blink_counter = 0
        self.last_action_time = 0.0
        self.last_blink_action = None
        
        # Iris detection
        self.bilateral_d = BILATERAL_D
        self.bilateral_sigma = BILATERAL_SIGMA
        self.iris_threshold = IRIS_THRESHOLD
        
        # Camera
        self.mirror_frame = MIRROR_FRAME
        self.cam_width = CAM_WIDTH
        self.cam_height = CAM_HEIGHT
        
        # Calibration thresholds
        self.samples_per_point = SAMPLES_PER_CALIBRATION_POINT
        self.min_samples_valid = MIN_SAMPLES_FOR_VALID_CALIBRATION
        self.min_cal_points = MIN_CALIBRATION_POINTS_FOR_ACCURACY

        # Mouth toggle state
        self.mouth_open_thr = MOUTH_OPEN_THR
        self.mouth_close_thr = MOUTH_CLOSE_THR
        self.mouth_toggle_cooldown = MOUTH_TOGGLE_COOLDOWN
        self.mouth_hyst = MOUTH_HYSTERESIS
        self.mouth_was_closed = True
        self.last_mouth_toggle_time = 0.0

        # LIVE sensitivity controls (NEW)
        self.edge_gamma = EDGE_GAMMA
        self.sens_gain = SENSITIVITY_GAIN
        self.snap = SNAP_MOVE

        print("✅ ULTIMATE DLIB Eye Mouse - Time Blink + Mouth Toggle + Live Sensitivity")
        print()

    # -------------------- Geometry helpers --------------------
    @staticmethod
    def _dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    # -------------------- Mouth helpers -----------------------
    def mouth_inner_ratio(self, landmarks):
        try:
            p60 = (landmarks.part(60).x, landmarks.part(60).y)
            p64 = (landmarks.part(64).x, landmarks.part(64).y)
            p62 = (landmarks.part(62).x, landmarks.part(62).y)
            p66 = (landmarks.part(66).x, landmarks.part(66).y)
            horiz = self._dist(p60, p64)
            vert  = self._dist(p62, p66)
            if horiz <= 1e-6:
                return 0.0
            return vert / horiz
        except Exception:
            return 0.0

    def update_mouth_toggle(self, mar_now):
        t = time.time()
        if mar_now < (self.mouth_close_thr - self.mouth_hyst):
            self.mouth_was_closed = True
            return
        if self.mouth_was_closed and mar_now >= self.mouth_close_thr:
            if (t - self.last_mouth_toggle_time) >= self.mouth_toggle_cooldown:
                self.mouse_active = not self.mouse_active
                self.last_mouth_toggle_time = t
                status = "STARTED ✅" if self.mouse_active else "STOPPED ⏸"
                print(f"👄 Mouth RE-ARM/OPEN → Mouse {status}")
            self.mouth_was_closed = False

    # -------------------- Eye/Iris helpers --------------------
    def detect_iris_center(self, frame, eye_points):
        try:
            eye_pts = np.array(eye_points)
            x_min = max(0, int(np.min(eye_pts[:, 0])) - 5)
            x_max = min(frame.shape[1], int(np.max(eye_pts[:, 0])) + 5)
            y_min = max(0, int(np.min(eye_pts[:, 1])) - 5)
            y_max = min(frame.shape[0], int(np.max(eye_pts[:, 1])) + 5)
            eye_region = frame[y_min:y_max, x_min:x_max].copy()
            if eye_region.size == 0:
                return None
            gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            bilateral = cv2.bilateralFilter(blurred, self.bilateral_d, self.bilateral_sigma, self.bilateral_sigma)
            equalized = cv2.equalizeHist(bilateral)
            _, iris_binary = cv2.threshold(equalized, self.iris_threshold, 255, cv2.THRESH_BINARY_INV)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            iris_binary = cv2.morphologyEx(iris_binary, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(iris_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            if area < 20 or area > 5000:
                return None
            m = cv2.moments(largest_contour)
            if m['m00'] == 0:
                return None
            iris_x_local = m['m10'] / m['m00']
            iris_y_local = m['m01'] / m['m00']
            iris_x = x_min + iris_x_local
            iris_y = y_min + iris_y_local
            return iris_x, iris_y
        except:
            return None
    
    def get_eye_aspect_ratio(self, eye_points):
        try:
            p2_p6 = math.hypot(eye_points[1][0] - eye_points[5][0],
                               eye_points[1][1] - eye_points[5][1])
            p3_p5 = math.hypot(eye_points[2][0] - eye_points[4][0],
                               eye_points[2][1] - eye_points[4][1])
            p1_p4 = math.hypot(eye_points[0][0] - eye_points[3][0],
                               eye_points[0][1] - eye_points[3][1])
            if p1_p4 == 0:
                return 0.3
            return (p2_p6 + p3_p5) / (2.0 * p1_p4)
        except:
            return 0.3

    # -------------------- Blink logic (TIME-BASED) -------------
    def process_blink(self, avg_ear):
        if avg_ear < self.ear_threshold:
            if not self.blink_started:
                self.blink_started = True
                self.blink_start_time = time.time()
                self.blink_counter = 0
            self.blink_counter += 1
            return None

        if self.blink_started:
            dur = time.time() - self.blink_start_time
            frames = self.blink_counter
            self.blink_started = False
            self.blink_counter = 0
            if frames >= self.short_min_frames:
                now = time.time()
                if (now - self.last_action_time) >= self.blink_refractory_sec:
                    if dur >= self.long_min_sec:
                        if self.mouse_active:
                            pyautogui.doubleClick()
                            print(f"👁️ LONG BLINK ({dur:.2f}s) → DOUBLE CLICK ✓✓")
                        self.last_action_time = now
                        self.last_blink_action = "long"
                        return "long"
                    else:
                        if self.mouse_active:
                            pyautogui.click()
                            print(f"👁️ SHORT BLINK ({dur:.2f}s) → SINGLE CLICK ✓")
                        self.last_action_time = now
                        self.last_blink_action = "short"
                        return "short"
        return None

    # -------------------- Calibration --------------------------
    def calibrate_5_point(self):
        self.cal_left_iris = {}
        self.cal_right_iris = {}
        self.cal_screen_points = {}
        self.calibration_count = 0
        
        print("\n" + "=" * 70)
        print("🎯 STARTING 5-POINT CALIBRATION")
        print("=" * 70)
        print(f"Hold SPACE to collect {self.samples_per_point} samples per point")

        cv2.namedWindow('5-Point Calibration', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('5-Point Calibration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        for point_num in range(1, 6):
            norm_x, norm_y, label = self.CALIBRATION_POINTS[point_num]
            target_screen_x = int(norm_x * self.screen_w)
            target_screen_y = int(norm_y * self.screen_h)
            print(f"\n📍 Calibrating Point {point_num}/5: {label} @ ({target_screen_x},{target_screen_y})")
            
            samples_left, samples_right = [], []
            collecting = False
            sample_count = 0
            target_samples = self.samples_per_point
            
            while sample_count < target_samples:
                ret, frame = cap.read()
                if not ret:
                    continue
                if self.mirror_frame:
                    frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.detector(gray, 0)
                display_frame = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
                face_found = False
                if faces:
                    face = faces[0]
                    landmarks = self.predictor(gray, face)
                    left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in self.LEFT_EYE_POINTS]
                    right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in self.RIGHT_EYE_POINTS]
                    face_found = True
                    if collecting:
                        left_iris = self.detect_iris_center(frame, left_eye)
                        right_iris = self.detect_iris_center(frame, right_eye)
                        if left_iris and right_iris:
                            samples_left.append(left_iris)
                            samples_right.append(right_iris)
                            sample_count += 1

                for p_num in range(1, 6):
                    p_norm_x, p_norm_y, _p_label = self.CALIBRATION_POINTS[p_num]
                    p_screen_x = int(p_norm_x * self.screen_w)
                    p_screen_y = int(p_norm_y * self.screen_h)
                    if p_num == point_num:
                        cv2.circle(display_frame, (p_screen_x, p_screen_y), 60, (0, 0, 255), -1)
                        cv2.circle(display_frame, (p_screen_x, p_screen_y), 60, (0, 255, 0), 3)
                    elif p_num in self.cal_screen_points:
                        cv2.circle(display_frame, (p_screen_x, p_screen_y), 20, (0, 255, 0), -1)
                    else:
                        cv2.circle(display_frame, (p_screen_x, p_screen_y), 20, (200, 200, 200), -1)
                    cv2.putText(display_frame, str(p_num), (p_screen_x - 10, p_screen_y + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

                cv2.putText(display_frame, f"Point {point_num}/5: {label}", 
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
                if not face_found:
                    cv2.putText(display_frame, "NO FACE DETECTED - Position your face in camera",
                                (50, self.screen_h - 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                else:
                    msg = "Hold SPACE to collect samples" if not collecting else f"Collecting... {sample_count}/{target_samples}"
                    cv2.putText(display_frame, msg, (50, self.screen_h - 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)

                cv2.putText(display_frame, "RED = Current | GREEN = Done | WHITE = Todo",
                            (50, self.screen_h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
                cv2.imshow('5-Point Calibration', display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    if not collecting:
                        collecting = True
                        print(f"   Collecting samples for point {point_num}...")
                elif key == ord('q'):
                    print("❌ Calibration cancelled")
                    cv2.destroyWindow('5-Point Calibration')
                    return False
                else:
                    if collecting and key != 255:
                        break
                if sample_count >= target_samples:
                    break

            if len(samples_left) < self.min_samples_valid or len(samples_right) < self.min_samples_valid:
                print(f"❌ Not enough samples for point {point_num} ({len(samples_left)}/{self.min_samples_valid})")
                print(f"   Try again for point {point_num}")
                continue

            avg_left = np.mean(samples_left, axis=0)
            avg_right = np.mean(samples_right, axis=0)
            self.cal_left_iris[point_num] = avg_left
            self.cal_right_iris[point_num] = avg_right
            self.cal_screen_points[point_num] = (norm_x, norm_y)
            self.calibration_count += 1
            print(f"✅ Point {point_num} calibrated ({len(samples_left)} samples)")
        
        cv2.destroyWindow('5-Point Calibration')
        if self.calibration_count >= self.min_cal_points:
            self.is_calibrated = True
            print("\n" + "=" * 70)
            print(f"✅ CALIBRATION COMPLETE! - {self.calibration_count}/5 points calibrated")
            print("=" * 70)
            print("\nBLINK CONTROLS:")
            print("  SHORT BLINK (quick) → SINGLE CLICK ✓")
            print("  LONG BLINK (hold)  → DOUBLE CLICK ✓✓")
            print("\nUse 1–5 to recalibrate any single point.")
            return True
        else:
            print(f"\n❌ Calibration failed - only {self.calibration_count}/{self.min_cal_points} points collected")
            return False

    def recalibrate_single_point(self, point_num):
        if point_num not in self.CALIBRATION_POINTS:
            return False
        norm_x, norm_y, label = self.CALIBRATION_POINTS[point_num]
        print(f"\n🔄 RECALIBRATING Point {point_num}: {label}  (hold SPACE)")

        cv2.namedWindow('Recalibrate Point', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Recalibrate Point', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        samples_left, samples_right = [], []
        collecting = False
        sample_count = 0
        target_samples = self.samples_per_point
        
        while sample_count < target_samples:
            ret, frame = cap.read()
            if not ret:
                continue
            if self.mirror_frame:
                frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray, 0)
            display_frame = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
            face_found = False
            if faces:
                face = faces[0]
                landmarks = self.predictor(gray, face)
                left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in self.LEFT_EYE_POINTS]
                right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in self.RIGHT_EYE_POINTS]
                face_found = True
                if collecting:
                    left_iris = self.detect_iris_center(frame, left_eye)
                    right_iris = self.detect_iris_center(frame, right_eye)
                    if left_iris and right_iris:
                        samples_left.append(left_iris)
                        samples_right.append(right_iris)
                        sample_count += 1

            for p_num in range(1, 6):
                p_norm_x, p_norm_y, _p_label = self.CALIBRATION_POINTS[p_num]
                p_screen_x = int(p_norm_x * self.screen_w)
                p_screen_y = int(p_norm_y * self.screen_h)
                if p_num == point_num:
                    cv2.circle(display_frame, (p_screen_x, p_screen_y), 60, (0, 0, 255), -1)
                    cv2.circle(display_frame, (p_screen_x, p_screen_y), 60, (0, 255, 0), 3)
                else:
                    cv2.circle(display_frame, (p_screen_x, p_screen_y), 20, (200, 200, 200), -1)
                cv2.putText(display_frame, str(p_num), (p_screen_x - 10, p_screen_y + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            cv2.putText(display_frame, f"Recalibrating Point {point_num}: {label}",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
            if not face_found:
                cv2.putText(display_frame, "NO FACE DETECTED",
                            (50, self.screen_h - 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            else:
                msg = f"Collecting... {sample_count}/{target_samples}" if collecting else "Hold SPACE to collect samples"
                cv2.putText(display_frame, msg, (50, self.screen_h - 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)

            cv2.imshow('Recalibrate Point', display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                if not collecting:
                    collecting = True
            elif key == ord('q'):
                cv2.destroyWindow('Recalibrate Point')
                return False
            else:
                if collecting and key != 255:
                    break
            if sample_count >= target_samples:
                break
        
        cv2.destroyWindow('Recalibrate Point')
        if len(samples_left) >= self.min_samples_valid and len(samples_right) >= self.min_samples_valid:
            avg_left = np.mean(samples_left, axis=0)
            avg_right = np.mean(samples_right, axis=0)
            self.cal_left_iris[point_num] = avg_left
            self.cal_right_iris[point_num] = avg_right
            self.cal_screen_points[point_num] = (norm_x, norm_y)
            print(f"✅ Point {point_num} recalibrated ({len(samples_left)} samples)")
            return True
        else:
            print("❌ Not enough samples for recalibration")
            return False

    # -------------------- Mapping / smoothing ------------------
    def gaze_to_screen(self, left_iris, right_iris):
        if not self.is_calibrated or not left_iris or not right_iris:
            return self.screen_w // 2, self.screen_h // 2
        try:
            avg_iris = np.array([(left_iris[0] + right_iris[0]) / 2,
                                 (left_iris[1] + right_iris[1]) / 2])
            cal_points = []
            cal_screens = []
            for p_num in sorted(self.cal_screen_points.keys()):
                cal_left = self.cal_left_iris[p_num]
                cal_right = self.cal_right_iris[p_num]
                cal_avg = (cal_left + cal_right) / 2
                cal_points.append(cal_avg)
                cal_screens.append(self.cal_screen_points[p_num])
            cal_points = np.array(cal_points)
            cal_screens = np.array(cal_screens)
            distances = np.linalg.norm(cal_points - avg_iris, axis=1)
            sorted_idx = np.argsort(distances)
            closest_idx = sorted_idx[:2]
            closest_dist = distances[closest_idx]
            if closest_dist[0] < 1e-6:
                u, v = cal_screens[closest_idx[0]]
            else:
                w = 1.0 / (closest_dist + 1e-6)
                w /= w.sum()
                uv = (cal_screens[closest_idx] * w[:, np.newaxis]).sum(axis=0)
                u, v = float(uv[0]), float(uv[1])

            # ---------- SENSITIVITY SHAPING (NEW): gamma + gain ----------
            ux = 2.0 * u - 1.0
            vy = 2.0 * v - 1.0
            g = max(0.30, min(1.50, self.edge_gamma))
            ux = np.sign(ux) * (abs(ux) ** g)
            vy = np.sign(vy) * (abs(vy) ** g)
            gain = max(0.5, min(2.5, self.sens_gain))
            ux *= gain
            vy *= gain
            ux = max(-1.0, min(1.0, ux))
            vy = max(-1.0, min(1.0, vy))
            u = 0.5 * (ux + 1.0)
            v = 0.5 * (vy + 1.0)
            # -------------------------------------------------------------

            screen_x = int(u * (self.screen_w - 1))
            screen_y = int(v * (self.screen_h - 1))
            screen_x = max(0, min(self.screen_w - 1, screen_x))
            screen_y = max(0, min(self.screen_h - 1, screen_y))
            return screen_x, screen_y
        except:
            return self.screen_w // 2, self.screen_h // 2
    
    def smooth_position(self, x, y):
        self.mouse_history.append([x, y])
        if len(self.mouse_history) > self.history_size:
            self.mouse_history.pop(0)
        avg_x = int(np.mean([pos[0] for pos in self.mouse_history]))
        avg_y = int(np.mean([pos[1] for pos in self.mouse_history]))
        return avg_x, avg_y


def main():
    global cap
    eye_mouse = UltimateDlibEyeMouse()
    
    # Initialize camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, eye_mouse.cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, eye_mouse.cam_height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("\n" + "=" * 70)
    print("🎮 ULTIMATE DLIB EYE MOUSE - Time Blink + Mouth Toggle + Live Sensitivity")
    print("=" * 70)
    print("\nKEYBOARD CONTROLS:")
    print("  C - Start full 5-point calibration")
    print("  1-5 - Recalibrate individual points")
    print("  S - Start/Stop Mouse Control (manual)")
    print("  + / - - Adjust Iris Threshold")
    print("  [ / ] - EDGE_GAMMA (lower = more sensitive corners)")
    print("  ; / ' - SENSITIVITY_GAIN (higher = larger movement)")
    print("  - / = - MOUSE smoothing window (smaller = snappier)")
    print("  0     - SNAP mouse move (duration=0) toggle")
    print("  R - Reset all calibration")
    print("  Q - Quit")
    print("\n👁️  BLINK CONTROLS:")
    print(f"  SHORT BLINK (< {eye_mouse.long_min_sec:.2f}s) → SINGLE CLICK ✓")
    print(f"  LONG BLINK (≥ {eye_mouse.long_min_sec:.2f}s) → DOUBLE CLICK ✓✓")
    print("\n👄  MOUTH CONTROL:")
    print("  Crossing into RE-ARM/OPEN (MAR ≥ close_thr) → toggle mouse ON/OFF")
    print("  Closing (MAR < close_thr - hysteresis) only re-arms (no toggle)")
    print("=" * 70 + "\n")
    
    def draw_info_panel(img, face_rect, avg_ear_value, mar_value):
        # LEFT panel only. If face overlaps left band, skip drawing (you already have bottom HUD).
        h, w = img.shape[:2]
        panel_w = 260
        if face_rect is not None and face_rect.left() < panel_w:
            return  # don't draw anywhere if left panel would cover the face

        x0 = 0  # left edge

        # Semi-transparent background
        overlay = img.copy()
        cv2.rectangle(overlay, (x0, 0), (x0 + panel_w, h), (0, 0, 0), -1)
        img[:] = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)

        # Text helpers
        x = x0 + 12
        y = 24
        dy = 20
        def line(txt, col=(0,255,0)):
            nonlocal y
            cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
            y += dy

        # Top status
        line(f"Mouse: {'ACTIVE' if eye_mouse.mouse_active else 'STOPPED'}",
             (0,255,0) if eye_mouse.mouse_active else (0,0,255))
        line(f"Calib: {'YES' if eye_mouse.is_calibrated else 'NO'} ({eye_mouse.calibration_count}/5)",
             (255,255,255))
        line(f"EAR: {avg_ear_value:.3f}  Thr:{eye_mouse.ear_threshold:.2f}", (255,255,255))
        line(f"MAR: {mar_value:.3f}  Open:{eye_mouse.mouth_open_thr:.2f}", (255,255,0))
        line(f"ReArm Thr:{eye_mouse.mouth_close_thr:.2f}  Hys:{eye_mouse.mouth_hyst:.2f}", (255,255,0))
        line(f"Mouth Cooldown: {eye_mouse.mouth_toggle_cooldown:.2f}s", (200,200,200))
        y += 6

        # Tunables
        line("— Parameters —", (180,255,180))
        line(f"IrisThr: {eye_mouse.iris_threshold}", (255,255,255))
        line(f"EdgeGamma: {eye_mouse.edge_gamma:.2f}", (255,255,255))
        line(f"Gain: {eye_mouse.sens_gain:.2f}", (255,255,255))
        line(f"SmoothN: {eye_mouse.history_size}", (255,255,255))
        line(f"SNAP: {'ON' if eye_mouse.snap else 'OFF'}", (255,255,255))
        y += 6

        # Blink parameters
        line("— Blink —", (180,255,180))
        line(f"ShortMinFrames: {eye_mouse.short_min_frames}", (200,200,200))
        line(f"LongBlink>= {eye_mouse.long_min_sec:.2f}s", (200,200,200))
        line(f"Refractory: {eye_mouse.blink_refractory_sec:.2f}s", (200,200,200))

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        if eye_mouse.mirror_frame:
            frame = cv2.flip(frame, 1)
        frame_h, frame_w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = eye_mouse.detector(gray, 0)
        display_frame = frame.copy()
        
        face_rect_for_panel = None
        avg_ear = 0.3
        mar = 0.0

        if faces:
            face = faces[0]
            face_rect_for_panel = face
            landmarks = eye_mouse.predictor(gray, face)
            left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in eye_mouse.LEFT_EYE_POINTS]
            right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in eye_mouse.RIGHT_EYE_POINTS]
            
            left_iris = eye_mouse.detect_iris_center(frame, left_eye)
            right_iris = eye_mouse.detect_iris_center(frame, right_eye)
            
            if eye_mouse.mouse_active and eye_mouse.is_calibrated and left_iris and right_iris:
                screen_x, screen_y = eye_mouse.gaze_to_screen(left_iris, right_iris)
                smooth_x, smooth_y = eye_mouse.smooth_position(screen_x, screen_y)
                pyautogui.moveTo(smooth_x, smooth_y, duration=0 if eye_mouse.snap else 0.01)
            
            left_ear = eye_mouse.get_eye_aspect_ratio(left_eye)
            right_ear = eye_mouse.get_eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2
            _ = eye_mouse.process_blink(avg_ear)

            mar = eye_mouse.mouth_inner_ratio(landmarks)
            eye_mouse.update_mouth_toggle(mar)

            # Draw eye landmarks
            for pt in left_eye:
                cv2.circle(display_frame, pt, 2, (0, 255, 0), -1)
            for pt in right_eye:
                cv2.circle(display_frame, pt, 2, (0, 255, 0), -1)
            # Draw detected iris centers
            if left_iris:
                cv2.circle(display_frame, (int(left_iris[0]), int(left_iris[1])), 3, (255, 0, 0), -1)
            if right_iris:
                cv2.circle(display_frame, (int(right_iris[0]), int(right_iris[1])), 3, (255, 0, 0), -1)

            # Draw mouth INNER landmark points (60–67)
            for i in eye_mouse.MOUTH_INNER_POINTS:
                cv2.circle(display_frame, (landmarks.part(i).x, landmarks.part(i).y), 2, (0, 255, 255), -1)
            
            # Small status near top (kept)
            blink_status_color = (0, 255, 0) if avg_ear > eye_mouse.ear_threshold else (0, 0, 255)
            blink_status_text = "Eyes Open ✓" if avg_ear > eye_mouse.ear_threshold else "Eyes Closed"
            cv2.putText(display_frame, blink_status_text, (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, blink_status_color, 2)
            cv2.putText(display_frame, f"EAR: {avg_ear:.3f}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if mar >= eye_mouse.mouth_open_thr:
                mouth_state = "OPEN"
                mouth_col = (0, 255, 0)
            elif mar >= eye_mouse.mouth_close_thr:
                mouth_state = "RE-ARMING (Toggles)"
                mouth_col = (0, 255, 255)
            else:
                mouth_state = "CLOSED"
                mouth_col = (200, 200, 200)
            cv2.putText(display_frame, f"Mouth: {mouth_state} | MAR: {mar:.3f}",
                        (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mouth_col, 2)

        else:
            cv2.putText(display_frame, "No face detected - position your face in view", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # LEFT info panel only (skips if it would cover face)
        draw_info_panel(display_frame, face_rect_for_panel, avg_ear, mar)

        # Existing top labels (kept)
        status_color = (0, 255, 0) if eye_mouse.mouse_active else (0, 0, 255)
        status_text = "ACTIVE ✅" if eye_mouse.mouse_active else "STOPPED ⏸"
        cv2.putText(display_frame, f"Mouse: {status_text}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        cal_text = f"YES ({eye_mouse.calibration_count}/5)" if eye_mouse.is_calibrated else "NO"
        cv2.putText(display_frame, f"Calibrated: {cal_text}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display_frame, f"IrisThr:{eye_mouse.iris_threshold}  "
                                   f"Gamma:{eye_mouse.edge_gamma:.2f}  "
                                   f"Gain:{eye_mouse.sens_gain:.2f}  "
                                   f"SmoothN:{eye_mouse.history_size}  "
                                   f"SNAP:{'ON' if eye_mouse.snap else 'OFF'}",
                    (20, frame_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('DLIB Eye Mouse - Sensitivity Controls', display_frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('c'):
            eye_mouse.mouse_active = False
            eye_mouse.calibrate_5_point()
        elif key >= ord('1') and key <= ord('5'):
            if eye_mouse.is_calibrated:
                point_num = int(chr(key))
                eye_mouse.mouse_active = False
                eye_mouse.recalibrate_single_point(point_num)
            else:
                print("❌ Calibrate first (press C)")
        elif key == ord('s'):
            if eye_mouse.is_calibrated:
                eye_mouse.mouse_active = not eye_mouse.mouse_active
                status = "STARTED ✅" if eye_mouse.mouse_active else "STOPPED ⏸"
                print(f"🎮 Mouse control {status}")
            else:
                print("❌ Calibrate first (press C)")
        elif key == ord('r'):
            eye_mouse.is_calibrated = False
            eye_mouse.mouse_active = False
            eye_mouse.cal_left_iris = {}
            eye_mouse.cal_right_iris = {}
            eye_mouse.cal_screen_points = {}
            eye_mouse.calibration_count = 0
            print("🔄 All calibration reset")
        elif key == ord('+') or key == ord('='):
            if key == ord('+'):
                eye_mouse.iris_threshold = min(100, eye_mouse.iris_threshold + 5)
                print(f"👁️ Iris threshold: {eye_mouse.iris_threshold}")
        elif key == ord('-'):
            eye_mouse.iris_threshold = max(20, eye_mouse.iris_threshold - 5)
            print(f"👁️ Iris threshold: {eye_mouse.iris_threshold}")

        # LIVE sensitivity hotkeys
        if key == ord('['):
            eye_mouse.edge_gamma = max(0.40, eye_mouse.edge_gamma - 0.05)
            print(f"🧭 EDGE_GAMMA: {eye_mouse.edge_gamma:.2f} (lower = more sensitive edges)")
        elif key == ord(']'):
            eye_mouse.edge_gamma = min(1.50, eye_mouse.edge_gamma + 0.05)
            print(f"🧭 EDGE_GAMMA: {eye_mouse.edge_gamma:.2f}")
        elif key == ord(';'):
            eye_mouse.sens_gain = min(2.50, eye_mouse.sens_gain + 0.10)
            print(f"⚡ SENSITIVITY_GAIN: {eye_mouse.sens_gain:.2f}")
        elif key == ord('\''):
            eye_mouse.sens_gain = max(0.50, eye_mouse.sens_gain - 0.10)
            print(f"⚡ SENSITIVITY_GAIN: {eye_mouse.sens_gain:.2f}")

        if key == ord(','):
            eye_mouse.history_size = max(1, eye_mouse.history_size - 1)
            print(f"🪄 MOUSE_HISTORY_SIZE: {eye_mouse.history_size}")
        elif key == ord('.'):
            eye_mouse.history_size = min(20, eye_mouse.history_size + 1)
            print(f"🪄 MOUSE_HISTORY_SIZE: {eye_mouse.history_size}")

        elif key == ord('0'):
            eye_mouse.snap = not eye_mouse.snap
            print(f"🚀 SNAP movement: {'ON' if eye_mouse.snap else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()
