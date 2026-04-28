# “””
OAK-D Pro — IR Dot Depth Point Cloud Studio

핵심 동작 원리:

1. IR 이미지에서 Blob Detection으로 각 IR dot 위치(u, v) 검출
1. depth frame에서 해당 (u, v) 위치의 depth 값 조회
1. depth가 유효한 dot → 3D 좌표 (X, Y, Z) 변환 → Point Cloud
1. depth가 0 또는 범위 외인 dot → Fail Spot

저장 파일:

- ir_left_<ts>.png   / ir_right_<ts>.png  : IR 카메라 원본
- depth_<ts>.png                           : 컬러맵 Depth 이미지
- pointcloud_<ts>.ply                      : dot 기반 Point Cloud
- report_<ts>.txt                          : Fail Spot Ratio 리포트

의존성:
pip install depthai opencv-python numpy open3d Pillow
“””

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
import depthai as dai
DEPTHAI_AVAILABLE = True
except ImportError:
DEPTHAI_AVAILABLE = False

try:
import open3d as o3d
OPEN3D_AVAILABLE = True
except ImportError:
OPEN3D_AVAILABLE = False

try:
from PIL import Image, ImageTk, ImageDraw, ImageFont
PIL_AVAILABLE = True
except ImportError:
PIL_AVAILABLE = False

# ── 테마 ─────────────────────────────────────────────────────────────────────

C = {
“bg”:       “#080b10”,
“panel”:    “#0d1117”,
“card”:     “#111827”,
“border”:   “#1f2937”,
“accent”:   “#22d3ee”,
“warn”:     “#f97316”,
“ok”:       “#4ade80”,
“fail”:     “#f43f5e”,
“text”:     “#e2e8f0”,
“mute”:     “#4b5563”,
“dot_ok”:   “#22d3ee”,
“dot_fail”: “#f43f5e”,
}

COLORMAP_OPTIONS = {
“TURBO”:   cv2.COLORMAP_TURBO,
“JET”:     cv2.COLORMAP_JET,
“MAGMA”:   cv2.COLORMAP_MAGMA,
“PLASMA”:  cv2.COLORMAP_PLASMA,
“VIRIDIS”: cv2.COLORMAP_VIRIDIS,
“HOT”:     cv2.COLORMAP_HOT,
}

# ════════════════════════════════════════════════════════════════════════════

# IR Dot Detector

# ════════════════════════════════════════════════════════════════════════════

class IRDotDetector:
“”“IR 이미지에서 dot 위치를 검출하고 depth 값을 조회”””

```
def __init__(self):
    self.blob_params = cv2.SimpleBlobDetector_Params()
    self.blob_params.filterByArea       = True
    self.blob_params.minArea            = 3
    self.blob_params.maxArea            = 300
    self.blob_params.filterByCircularity = True
    self.blob_params.minCircularity     = 0.4
    self.blob_params.filterByConvexity  = True
    self.blob_params.minConvexity       = 0.6
    self.blob_params.filterByInertia    = True
    self.blob_params.minInertiaRatio    = 0.2
    self.blob_params.minThreshold       = 100
    self.blob_params.maxThreshold       = 255
    self.blob_params.thresholdStep      = 10
    self.detector = cv2.SimpleBlobDetector_create(self.blob_params)

def detect(self, ir_frame: np.ndarray,
           depth_frame: np.ndarray,
           intrinsics: dict,
           min_depth: int = 100,
           max_depth: int = 5000,
           thresh_val: int = 120) -> dict:
    """
    Returns:
        {
          "keypoints":    [cv2.KeyPoint, ...],   # 검출된 모든 dot
          "pts_3d_ok":    [(x,y,z), ...],        # 유효 depth dot의 3D 좌표
          "pts_3d_fail":  [(u,v), ...],           # 실패 dot의 2D 위치
          "total":        int,
          "ok":           int,
          "fail":         int,
          "fail_ratio":   float,                  # 0.0 ~ 1.0
          "overlay":      np.ndarray,             # 시각화용 BGR
        }
    """
    fx = intrinsics["fx"]; fy = intrinsics["fy"]
    cx = intrinsics["cx"]; cy = intrinsics["cy"]

    # ── 전처리: 밝은 점 강조 ──
    if ir_frame.dtype != np.uint8:
        norm = cv2.normalize(ir_frame, None, 0, 255, cv2.NORM_MINMAX)
        gray = norm.astype(np.uint8)
    else:
        gray = ir_frame.copy()

    # 밝기 균일화
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 임계값으로 bright dot 마스크
    _, bright_mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    # 노이즈 제거
    kernel = np.ones((3, 3), np.uint8)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)

    # ── Blob 검출 ──
    # SimpleBlobDetector는 어두운 blob을 찾으므로 반전
    inv = cv2.bitwise_not(bright_mask)
    keypoints = self.detector.detect(inv)

    # Blob이 너무 적으면 contour 기반 fallback
    if len(keypoints) < 10:
        keypoints = self._contour_fallback(bright_mask)

    # ── depth 조회 및 3D 변환 ──
    h_d, w_d = depth_frame.shape
    h_ir, w_ir = gray.shape
    scale_x = w_d / w_ir
    scale_y = h_d / h_ir

    pts_3d_ok   = []
    pts_3d_fail = []

    for kp in keypoints:
        u_ir, v_ir = kp.pt
        # depth frame 좌표로 스케일
        u_d = int(u_ir * scale_x)
        v_d = int(v_ir * scale_y)
        u_d = np.clip(u_d, 0, w_d - 1)
        v_d = np.clip(v_d, 0, h_d - 1)

        # 주변 3x3 median으로 depth noise 감소
        u0 = max(0, u_d - 1); u1 = min(w_d, u_d + 2)
        v0 = max(0, v_d - 1); v1 = min(h_d, v_d + 2)
        patch = depth_frame[v0:v1, u0:u1].flatten()
        valid_patch = patch[patch > 0]
        z_mm = int(np.median(valid_patch)) if len(valid_patch) > 0 else 0

        if min_depth <= z_mm <= max_depth:
            z = z_mm / 1000.0
            x = (u_ir - cx) * z / fx
            y = -(v_ir - cy) * z / fy   # Y축 반전 (화면 하→3D 상)
            pts_3d_ok.append((x, y, z, z_mm))
        else:
            pts_3d_fail.append((int(u_ir), int(v_ir)))

    total = len(keypoints)
    ok    = len(pts_3d_ok)
    fail  = len(pts_3d_fail)
    fail_ratio = fail / total if total > 0 else 0.0

    # ── 시각화 오버레이 ──
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # 성공 dot: 청록 원
    for (x, y, z, z_mm) in pts_3d_ok:
        u = int((x * fx / z) + cx) if z > 0 else 0
        v = int((-y * fy / z) + cy) if z > 0 else 0
        cv2.circle(overlay, (u, v), 4, (0, 220, 200), 1)
        cv2.circle(overlay, (u, v), 1, (0, 255, 220), -1)
    # 실패 dot: 빨간 X
    for (u, v) in pts_3d_fail:
        cv2.drawMarker(overlay, (u, v), (60, 60, 255),
                       cv2.MARKER_CROSS, 8, 1)

    return {
        "keypoints":   keypoints,
        "pts_3d_ok":   pts_3d_ok,
        "pts_3d_fail": pts_3d_fail,
        "total":       total,
        "ok":          ok,
        "fail":        fail,
        "fail_ratio":  fail_ratio,
        "overlay":     overlay,
        "bright_mask": bright_mask,
    }

def _contour_fallback(self, bright_mask):
    """Blob 검출 실패 시 contour 기반 dot 검출"""
    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    kps = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 2 < area < 400:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                size = np.sqrt(area)
                kps.append(cv2.KeyPoint(float(cx), float(cy), float(size)))
    return kps

def update_params(self, min_area=3, max_area=300, threshold=120):
    self.blob_params.minArea      = min_area
    self.blob_params.maxArea      = max_area
    self.blob_params.minThreshold = threshold
    self.detector = cv2.SimpleBlobDetector_create(self.blob_params)
```

# ════════════════════════════════════════════════════════════════════════════

# 카메라 백엔드

# ════════════════════════════════════════════════════════════════════════════

class OakDCamera:
def **init**(self):
self.device     = None
self.running    = False
self.frame_lock = threading.Lock()
self.frame_left  = None
self.frame_right = None
self.frame_depth = None
self.intrinsics  = None

```
def start(self, ir_brightness=800):
    pipeline = dai.Pipeline()

    mono_l = pipeline.create(dai.node.MonoCamera)
    mono_r = pipeline.create(dai.node.MonoCamera)
    mono_l.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    mono_r.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    mono_l.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_r.setBoardSocket(dai.CameraBoardSocket.CAM_C)

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(False)
    stereo.setSubpixel(False)   # subpixel OFF → depth 단위 mm 직접

    mono_l.out.link(stereo.left)
    mono_r.out.link(stereo.right)

    for name, src in [("left", stereo.rectifiedLeft),
                      ("right", stereo.rectifiedRight),
                      ("depth", stereo.depth)]:
        xo = pipeline.create(dai.node.XLinkOut)
        xo.setStreamName(name)
        src.link(xo.input)

    self.device = dai.Device(pipeline)
    self.device.setIrFloodLightBrightness(0)
    self.device.setIrLaserDotProjectorBrightness(ir_brightness)

    # 실제 depth 해상도로 intrinsics 요청
    time.sleep(0.5)
    calib = self.device.readCalibration()
    # depth frame 해상도 확인을 위해 첫 프레임 대기
    q_tmp = self.device.getOutputQueue("depth", 4, True)
    d_tmp = q_tmp.get().getFrame()
    h_d, w_d = d_tmp.shape
    M = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_C, w_d, h_d)
    self.intrinsics = {
        "fx": M[0][0], "fy": M[1][1],
        "cx": M[0][2], "cy": M[1][2],
        "w": w_d, "h": h_d,
    }

    self.running = True
    threading.Thread(target=self._loop, daemon=True).start()

def _loop(self):
    ql = self.device.getOutputQueue("left",  4, False)
    qr = self.device.getOutputQueue("right", 4, False)
    qd = self.device.getOutputQueue("depth", 4, False)
    while self.running:
        fl = ql.tryGet(); fr = qr.tryGet(); fd = qd.tryGet()
        with self.frame_lock:
            if fl: self.frame_left  = fl.getCvFrame()
            if fr: self.frame_right = fr.getCvFrame()
            if fd: self.frame_depth = fd.getFrame()
        time.sleep(0.01)

def stop(self):
    self.running = False
    time.sleep(0.2)
    if self.device:
        self.device.close()

def set_ir(self, v):
    if self.device:
        self.device.setIrLaserDotProjectorBrightness(int(v))

def get_frames(self):
    with self.frame_lock:
        return (
            self.frame_left.copy()  if self.frame_left  is not None else None,
            self.frame_right.copy() if self.frame_right is not None else None,
            self.frame_depth.copy() if self.frame_depth is not None else None,
        )
```

# ════════════════════════════════════════════════════════════════════════════

# Demo 백엔드

# ════════════════════════════════════════════════════════════════════════════

class DemoCamera:
def **init**(self):
self.running    = False
self.frame_lock = threading.Lock()
self.frame_left = self.frame_right = self.frame_depth = None
self.intrinsics = {“fx”: 450.0, “fy”: 450.0,
“cx”: 320.0, “cy”: 180.0, “w”: 640, “h”: 360}
self._t = 0.0

```
def start(self, ir_brightness=800):
    self.running = True
    threading.Thread(target=self._loop, daemon=True).start()

def stop(self): self.running = False
def set_ir(self, v): pass

def _loop(self):
    while self.running:
        self._t += 0.04
        h, w = 360, 640
        # IR dot 격자 생성
        ir = np.zeros((h, w), np.uint8)
        for dy in range(15, h, 25):
            for dx in range(15, w, 25):
                # 3D 구면 시뮬레이션 → 위치 왜곡
                nx = (dx - w/2) / (w/2)
                ny = (dy - h/2) / (h/2)
                r  = np.sqrt(nx**2 + ny**2)
                distort = 1 + 0.3 * r * np.sin(self._t * 0.5)
                px = int(dx + nx * 8 * distort)
                py = int(dy + ny * 8 * distort)
                if 0 <= px < w and 0 <= py < h:
                    cv2.circle(ir, (px, py), 2, 200, -1)

        # depth: 중앙 구면 시뮬레이션 (얼굴처럼)
        depth = np.zeros((h, w), np.uint16)
        for y in range(h):
            for x in range(0, w, 2):
                nx = (x - w/2) / (w/2)
                ny = (y - h/2) / (h/2)
                r  = np.sqrt(nx**2 + ny**2)
                z  = int(1000 + 300 * r**2 + 50 * np.sin(self._t + r*5))
                # 일부 fail 픽셀 (가장자리)
                if r > 0.9:
                    depth[y, x:x+2] = 0
                else:
                    depth[y, x:x+2] = z

        with self.frame_lock:
            self.frame_left  = ir
            self.frame_right = np.roll(ir, -12, axis=1)
            self.frame_depth = depth
        time.sleep(0.033)

def get_frames(self):
    with self.frame_lock:
        return (
            self.frame_left.copy()  if self.frame_left  is not None else None,
            self.frame_right.copy() if self.frame_right is not None else None,
            self.frame_depth.copy() if self.frame_depth is not None else None,
        )
```

# ════════════════════════════════════════════════════════════════════════════

# 유틸리티

# ════════════════════════════════════════════════════════════════════════════

def depth_colormap(depth, cmap_id, min_d=100, max_d=5000):
clipped = np.clip(depth.astype(np.float32), min_d, max_d)
norm    = ((clipped - min_d) / (max_d - min_d) * 255).astype(np.uint8)
colored = cv2.applyColorMap(norm, cmap_id)
colored[depth == 0] = 0
return colored

def cv_to_tk(frame, w, h):
if frame.ndim == 2:
rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
else:
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
img = Image.fromarray(cv2.resize(rgb, (w, h)))
return ImageTk.PhotoImage(img)

def save_ply(pts_3d_ok, path):
“”“유효 dot 3D 좌표를 PLY로 저장. depth에 따라 컬러 그라디언트.”””
if not pts_3d_ok:
return 0

```
arr = np.array([(x, y, z) for x, y, z, _ in pts_3d_ok])
z_vals = arr[:, 2]
z_min, z_max = z_vals.min(), z_vals.max() + 1e-6

if OPEN3D_AVAILABLE:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr)
    # depth 기반 컬러맵
    norm = (z_vals - z_min) / (z_max - z_min)
    lut  = cv2.applyColorMap(
        (norm * 255).astype(np.uint8).reshape(-1, 1),
        cv2.COLORMAP_TURBO).reshape(-1, 3)[:, ::-1] / 255.0
    pcd.colors = o3d.utility.Vector3dVector(lut)
    o3d.io.write_point_cloud(path, pcd)
else:
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(arr)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i, (x, y, z) in enumerate(arr):
            t   = (z_vals[i] - z_min) / (z_max - z_min)
            r   = int(min(255, t * 2 * 255))
            b   = int(min(255, (1 - t) * 2 * 255))
            g   = 128
            f.write(f"{x:.5f} {y:.5f} {z:.5f} {r} {g} {b}\n")
return len(arr)
```

def save_report(result: dict, ir_brightness: int, path: str, ts: str):
“”“Fail Spot Ratio 리포트 저장”””
lines = [
“=” * 50,
“ OAK-D Pro  IR Dot Depth Report”,
f” Timestamp : {ts}”,
“=” * 50,
f” IR Dot Projector  : {ir_brightness} mA”,
f” Total Dots Detected: {result[‘total’]}”,
f” Valid (OK) Dots    : {result[‘ok’]}”,
f” Failed Dots        : {result[‘fail’]}”,
f” Fail Spot Ratio    : {result[‘fail_ratio’]*100:.2f}%”,
“”,
“ Fail Spot 위치 (u, v) :”,
]
for u, v in result[“pts_3d_fail”]:
lines.append(f”   ({u:4d}, {v:4d})”)
lines += [””, “=” * 50]
with open(path, “w”, encoding=“utf-8”) as f:
f.write(”\n”.join(lines))

# ════════════════════════════════════════════════════════════════════════════

# GUI

# ════════════════════════════════════════════════════════════════════════════

PW, PH = 500, 282   # 프리뷰 캔버스 크기

class App(tk.Tk):
def **init**(self):
super().**init**()
self.title(“OAK-D Pro  ·  IR Dot Depth Point Cloud Studio”)
self.configure(bg=C[“bg”])
self.resizable(True, True)

```
    self.cam: OakDCamera | DemoCamera = None
    self._running = False
    self.detector  = IRDotDetector()
    self._last_result = None
    self._last_frames = (None, None, None)

    self.save_dir = str(Path.home() / "oakd_captures")
    os.makedirs(self.save_dir, exist_ok=True)

    # vars
    self.v_ir       = tk.IntVar(value=800)
    self.v_cmap     = tk.StringVar(value="TURBO")
    self.v_min_d    = tk.IntVar(value=150)
    self.v_max_d    = tk.IntVar(value=4000)
    self.v_thresh   = tk.IntVar(value=120)
    self.v_min_area = tk.IntVar(value=3)
    self.v_max_area = tk.IntVar(value=300)
    self.v_status   = tk.StringVar(value="카메라 연결 대기 중...")
    self.v_fps      = tk.StringVar(value="-- fps")
    self.v_dots     = tk.StringVar(value="dots: --")
    self.v_ok       = tk.StringVar(value="ok: --")
    self.v_fail     = tk.StringVar(value="fail: --")
    self.v_ratio    = tk.StringVar(value="fail ratio: --%")

    self.demo_mode = not DEPTHAI_AVAILABLE
    self._fps_t = time.time()
    self._fps_n = 0

    self._build()
    self.protocol("WM_DELETE_WINDOW", self._on_close)

# ── 레이아웃 ────────────────────────────────────────────────────────────
def _build(self):
    # 헤더
    hdr = tk.Frame(self, bg="#0a0e17", height=48)
    hdr.pack(fill="x")
    hdr.pack_propagate(False)
    tk.Label(hdr, text="◈ OAK-D PRO  IR DOT DEPTH STUDIO",
             font=("Courier New", 13, "bold"),
             fg=C["accent"], bg="#0a0e17").pack(side="left", padx=18, pady=10)
    if self.demo_mode:
        tk.Label(hdr, text="[ DEMO ]", font=("Courier New", 10),
                 fg=C["warn"], bg="#0a0e17").pack(side="left")
    tk.Label(hdr, textvariable=self.v_fps,
             font=("Courier New", 11), fg=C["warn"], bg="#0a0e17"
             ).pack(side="right", padx=18)
    tk.Frame(self, bg=C["border"], height=1).pack(fill="x")

    # 본문
    body = tk.Frame(self, bg=C["bg"])
    body.pack(fill="both", expand=True, padx=10, pady=8)

    left_col  = tk.Frame(body, bg=C["bg"])
    left_col.pack(side="left", fill="both", expand=True)
    right_col = tk.Frame(body, bg=C["panel"], width=270)
    right_col.pack(side="right", fill="y", padx=(8, 0))
    right_col.pack_propagate(False)

    self._build_previews(left_col)
    self._build_stats(left_col)
    self._build_controls(right_col)

    # 상태바
    sb = tk.Frame(self, bg=C["card"], height=24)
    sb.pack(fill="x", side="bottom")
    sb.pack_propagate(False)
    tk.Label(sb, textvariable=self.v_status,
             font=("Courier New", 8), fg=C["mute"], bg=C["card"],
             anchor="w").pack(side="left", padx=8)
    tk.Label(sb, text=f"save → {self.save_dir}",
             font=("Courier New", 8), fg=C["mute"], bg=C["card"],
             anchor="e").pack(side="right", padx=8)

def _build_previews(self, parent):
    row1 = tk.Frame(parent, bg=C["bg"])
    row1.pack(fill="x", pady=(0, 5))

    def cam_card(row, title, attr):
        f = tk.Frame(row, bg=C["card"],
                     highlightbackground=C["border"], highlightthickness=1)
        f.pack(side="left", expand=True, fill="both", padx=(0, 5))
        tk.Label(f, text=title, font=("Courier New", 8, "bold"),
                 fg=C["accent"], bg=C["card"]).pack(anchor="w", padx=8, pady=(5,2))
        cv = tk.Canvas(f, width=PW, height=PH, bg="#04060a",
                       bd=0, highlightthickness=0)
        cv.pack(padx=5, pady=(0, 5))
        cv.create_text(PW//2, PH//2, text="NO SIGNAL",
                       font=("Courier New", 12, "bold"),
                       fill=C["mute"], tags="ns")
        setattr(self, attr, cv)

    cam_card(row1, "◧  LEFT IR  (CAM_B)", "cv_left")
    cam_card(row1, "◨  RIGHT IR  (CAM_C)", "cv_right")

    row2 = tk.Frame(parent, bg=C["bg"])
    row2.pack(fill="x", pady=(0, 5))

    def big_card(row, title, attr, w):
        f = tk.Frame(row, bg=C["card"],
                     highlightbackground=C["border"], highlightthickness=1)
        f.pack(side="left", fill="both", expand=True, padx=(0, 5))
        tk.Label(f, text=title, font=("Courier New", 8, "bold"),
                 fg=C["warn"], bg=C["card"]).pack(anchor="w", padx=8, pady=(5,2))
        cv = tk.Canvas(f, width=w, height=PH, bg="#04060a",
                       bd=0, highlightthickness=0)
        cv.pack(padx=5, pady=(0, 5))
        cv.create_text(w//2, PH//2, text="NO SIGNAL",
                       font=("Courier New", 12, "bold"),
                       fill=C["mute"], tags="ns")
        setattr(self, attr, cv)

    big_card(row2, "▦  DEPTH MAP", "cv_depth", PW)
    big_card(row2, "◉  IR DOT OVERLAY  (● ok  ✕ fail)", "cv_overlay", PW)

    self._tk = {}  # PhotoImage refs

def _build_stats(self, parent):
    sf = tk.Frame(parent, bg=C["card"],
                  highlightbackground=C["border"], highlightthickness=1)
    sf.pack(fill="x", pady=(0, 5))

    inner = tk.Frame(sf, bg=C["card"])
    inner.pack(fill="x", padx=10, pady=6)

    def stat(frame, var, color, width=160):
        tk.Label(frame, textvariable=var,
                 font=("Courier New", 11, "bold"),
                 fg=color, bg=C["card"], width=18, anchor="w"
                 ).pack(side="left", padx=6)

    stat(inner, self.v_dots,  C["text"])
    stat(inner, self.v_ok,    C["ok"])
    stat(inner, self.v_fail,  C["fail"])

    ratio_lbl = tk.Label(inner, textvariable=self.v_ratio,
                         font=("Courier New", 13, "bold"),
                         fg=C["fail"], bg=C["card"])
    ratio_lbl.pack(side="left", padx=12)
    self._ratio_label = ratio_lbl

# ── 컨트롤 패널 ─────────────────────────────────────────────────────────
def _card_section(self, parent, title):
    outer = tk.Frame(parent, bg=C["card"],
                     highlightbackground=C["border"], highlightthickness=1)
    outer.pack(fill="x", padx=8, pady=(5, 0))
    tk.Label(outer, text=title, font=("Courier New", 8, "bold"),
             fg=C["mute"], bg=C["card"]).pack(anchor="w", padx=8, pady=(6,1))
    body = tk.Frame(outer, bg=C["card"])
    body.pack(fill="x", padx=8, pady=(0, 8))
    return body

def _slider_row(self, parent, label, var, lo, hi, cmd=None):
    tk.Label(parent, text=label, font=("Courier New", 8),
             fg=C["mute"], bg=C["card"]).pack(anchor="w")
    row = tk.Frame(parent, bg=C["card"])
    row.pack(fill="x")
    val_lbl = tk.Label(row, text=str(var.get()),
                       font=("Courier New", 9, "bold"),
                       fg=C["accent"], bg=C["card"], width=6)
    val_lbl.pack(side="right")
    def _update(v, lbl=val_lbl, variable=var):
        lbl.config(text=str(int(float(v))))
        if cmd: cmd(v)
    sc = ttk.Scale(row, from_=lo, to=hi, orient="horizontal",
                   variable=var, command=_update)
    sc.pack(side="left", fill="x", expand=True)

def _build_controls(self, parent):
    tk.Label(parent, text="CONTROLS", font=("Courier New", 10, "bold"),
             fg=C["accent"], bg=C["panel"]).pack(pady=(12, 4))

    # 카메라
    cc = self._card_section(parent, "● CAMERA")
    self.btn_cam = tk.Button(cc, text="▶  카메라 시작",
                             font=("Courier New", 10, "bold"),
                             bg=C["accent"], fg=C["bg"],
                             relief="flat", cursor="hand2",
                             padx=6, pady=6,
                             command=self._toggle_cam)
    self.btn_cam.pack(fill="x")

    # IR
    ir = self._card_section(parent, "◉ IR DOT PROJECTOR")
    self._slider_row(ir, "밝기 (mA)", self.v_ir, 0, 1200,
                     cmd=lambda v: self.cam.set_ir(int(float(v))) if self.cam else None)
    btn_row = tk.Frame(ir, bg=C["card"])
    btn_row.pack(fill="x", pady=(4,0))
    for lbl, val in [("OFF",0),("LOW",400),("MED",800),("MAX",1200)]:
        tk.Button(btn_row, text=lbl, font=("Courier New", 8, "bold"),
                  bg=C["border"], fg=C["text"], relief="flat",
                  cursor="hand2", padx=3, pady=2,
                  command=lambda v=val: [self.v_ir.set(v),
                                         self.cam.set_ir(v) if self.cam else None]
                  ).pack(side="left", expand=True, fill="x", padx=1)

    # Dot Detector
    dd = self._card_section(parent, "◎ DOT DETECTOR")
    self._slider_row(dd, "밝기 임계값 (Threshold)", self.v_thresh, 60, 240,
                     cmd=self._on_detector_change)
    self._slider_row(dd, "최소 Blob 면적 (px²)", self.v_min_area, 1, 50,
                     cmd=self._on_detector_change)
    self._slider_row(dd, "최대 Blob 면적 (px²)", self.v_max_area, 50, 800,
                     cmd=self._on_detector_change)

    # Depth
    dp = self._card_section(parent, "▦ DEPTH RANGE")
    self._slider_row(dp, "최소 거리 (mm)", self.v_min_d, 100, 2000)
    self._slider_row(dp, "최대 거리 (mm)", self.v_max_d, 500, 10000)

    tk.Label(dp, text="컬러맵", font=("Courier New", 8),
             fg=C["mute"], bg=C["card"]).pack(anchor="w", pady=(4,0))
    cb = ttk.Combobox(dp, textvariable=self.v_cmap,
                      values=list(COLORMAP_OPTIONS.keys()),
                      state="readonly", font=("Courier New", 8))
    cb.pack(fill="x")

    # 저장
    sv = self._card_section(parent, "▤ SAVE")

    def save_btn(text, color, cmd):
        tk.Button(sv, text=text, font=("Courier New", 8, "bold"),
                  bg=color, fg="white", relief="flat",
                  cursor="hand2", padx=4, pady=5,
                  command=cmd).pack(fill="x", pady=2)

    save_btn("📷  IR Left 저장",     "#0e7490", self._save_ir_left)
    save_btn("📷  IR Right 저장",    "#155e75", self._save_ir_right)
    save_btn("🎨  Depth Map 저장",   C["warn"],  self._save_depth)
    save_btn("☁   Point Cloud (PLY)", "#7c3aed", self._save_ply)
    save_btn("📋  전체 저장 + 리포트", C["ok"],    self._save_all)

    tk.Button(sv, text="📁 저장 폴더 변경",
              font=("Courier New", 7), bg=C["border"], fg=C["mute"],
              relief="flat", cursor="hand2", pady=3,
              command=self._change_dir).pack(fill="x", pady=(4,0))

# ── 카메라 제어 ─────────────────────────────────────────────────────────
def _toggle_cam(self):
    if not self._running:
        self.btn_cam.config(state="disabled", text="연결 중...")
        threading.Thread(target=self._start_cam, daemon=True).start()
    else:
        self._stop_cam()

def _start_cam(self):
    try:
        self.cam = DemoCamera() if self.demo_mode else OakDCamera()
        self.cam.start(self.v_ir.get())
        self._running = True
        self.after(0, self._on_ready)
    except Exception as e:
        self.after(0, lambda: messagebox.showerror("오류", str(e)))
        self.after(0, lambda: self.btn_cam.config(
            state="normal", text="▶  카메라 시작"))

def _on_ready(self):
    self.btn_cam.config(state="normal", text="■  카메라 중지",
                        bg=C["fail"], activebackground="#be123c")
    mode = "DEMO" if self.demo_mode else "OAK-D Pro"
    self.v_status.set(f"[{mode}] 연결됨")
    self._update_loop()

def _stop_cam(self):
    self._running = False
    if self.cam: self.cam.stop()
    self.btn_cam.config(text="▶  카메라 시작", bg=C["accent"])
    self.v_status.set("카메라 중지됨")

def _on_detector_change(self, _=None):
    self.detector.update_params(
        min_area  = self.v_min_area.get(),
        max_area  = self.v_max_area.get(),
        threshold = self.v_thresh.get(),
    )

# ── 메인 루프 ───────────────────────────────────────────────────────────
def _update_loop(self):
    if not self._running:
        return

    fl, fr, fd = self.cam.get_frames()
    self._last_frames = (fl, fr, fd)

    cmap_id = COLORMAP_OPTIONS[self.v_cmap.get()]
    min_d   = self.v_min_d.get()
    max_d   = self.v_max_d.get()

    if fl is not None:
        self._show(self.cv_left, fl, PW, PH, "left")

    if fr is not None:
        self._show(self.cv_right, fr, PW, PH, "right")

    if fd is not None:
        col = depth_colormap(fd, cmap_id, min_d, max_d)
        self._show(self.cv_depth, col, PW, PH, "depth")

    # Dot detection (left IR 기준)
    if fl is not None and fd is not None:
        result = self.detector.detect(
            fl, fd, self.cam.intrinsics,
            min_depth=min_d, max_depth=max_d,
            thresh_val=self.v_thresh.get())
        self._last_result = result

        self._show(self.cv_overlay, result["overlay"], PW, PH, "overlay")
        self._update_stats(result)

    # FPS
    self._fps_n += 1
    now = time.time()
    if now - self._fps_t >= 1.0:
        self.v_fps.set(f"{self._fps_n/(now-self._fps_t):.1f} fps")
        self._fps_n = 0; self._fps_t = now

    self.after(33, self._update_loop)

def _show(self, canvas, frame, w, h, key):
    tk_img = cv_to_tk(frame, w, h)
    self._tk[key] = tk_img
    canvas.delete("ns"); canvas.delete("img")
    canvas.create_image(0, 0, anchor="nw", image=tk_img, tags="img")

def _update_stats(self, result):
    t  = result["total"]
    ok = result["ok"]
    fa = result["fail"]
    r  = result["fail_ratio"] * 100

    self.v_dots.set(f"dots: {t}")
    self.v_ok.set(f"ok:   {ok}")
    self.v_fail.set(f"fail: {fa}")
    self.v_ratio.set(f"FAIL RATIO  {r:.1f}%")

    # 색상: 비율에 따라
    color = C["ok"] if r < 20 else C["warn"] if r < 50 else C["fail"]
    self._ratio_label.config(fg=color)
    self.v_status.set(
        f"dots={t}  ok={ok}  fail={fa}  fail_ratio={r:.1f}%  "
        f"| IR={self.v_ir.get()}mA")

# ── 저장 ────────────────────────────────────────────────────────────────
def _ts(self):
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _check_running(self):
    if not self._running:
        messagebox.showwarning("저장 오류", "카메라가 실행 중이 아닙니다.")
        return False
    return True

def _save_ir_left(self):
    if not self._check_running(): return
    fl, _, _ = self._last_frames
    if fl is None: return
    ts   = self._ts()
    path = os.path.join(self.save_dir, f"ir_left_{ts}.png")
    cv2.imwrite(path, fl)
    self.v_status.set(f"✓ IR Left 저장: {path}")

def _save_ir_right(self):
    if not self._check_running(): return
    _, fr, _ = self._last_frames
    if fr is None: return
    ts   = self._ts()
    path = os.path.join(self.save_dir, f"ir_right_{ts}.png")
    cv2.imwrite(path, fr)
    self.v_status.set(f"✓ IR Right 저장: {path}")

def _save_depth(self):
    if not self._check_running(): return
    _, _, fd = self._last_frames
    if fd is None: return
    ts   = self._ts()
    path = os.path.join(self.save_dir, f"depth_{ts}.png")
    col  = depth_colormap(fd, COLORMAP_OPTIONS[self.v_cmap.get()],
                           self.v_min_d.get(), self.v_max_d.get())
    cv2.imwrite(path, col)
    self.v_status.set(f"✓ Depth Map 저장: {path}")

def _save_ply(self):
    if not self._check_running(): return
    if not self._last_result or not self._last_result["pts_3d_ok"]:
        messagebox.showwarning("저장 오류", "유효한 dot 3D 포인트가 없습니다.\n"
                               "dot이 검출되고 있는지 확인하세요.")
        return
    ts   = self._ts()
    path = os.path.join(self.save_dir, f"pointcloud_{ts}.ply")
    self.v_status.set("PLY 저장 중...")
    self.update()

    result = self._last_result
    def _do():
        n = save_ply(result["pts_3d_ok"], path)
        self.after(0, lambda: self.v_status.set(
            f"✓ PLY 저장: {path}  ({n:,} points)"))
    threading.Thread(target=_do, daemon=True).start()

def _save_all(self):
    """IR 좌/우 + Depth + PLY + Report 한번에 저장"""
    if not self._check_running(): return
    fl, fr, fd = self._last_frames
    result = self._last_result
    ts = self._ts()

    saved = []
    if fl is not None:
        p = os.path.join(self.save_dir, f"ir_left_{ts}.png")
        cv2.imwrite(p, fl); saved.append("ir_left")
    if fr is not None:
        p = os.path.join(self.save_dir, f"ir_right_{ts}.png")
        cv2.imwrite(p, fr); saved.append("ir_right")
    if fd is not None:
        col = depth_colormap(fd, COLORMAP_OPTIONS[self.v_cmap.get()],
                              self.v_min_d.get(), self.v_max_d.get())
        p = os.path.join(self.save_dir, f"depth_{ts}.png")
        cv2.imwrite(p, col); saved.append("depth")
        # overlay
        if result:
            p2 = os.path.join(self.save_dir, f"overlay_{ts}.png")
            cv2.imwrite(p2, result["overlay"]); saved.append("overlay")

    if result:
        # report
        rp = os.path.join(self.save_dir, f"report_{ts}.txt")
        save_report(result, self.v_ir.get(), rp, ts)
        saved.append("report")

        if result["pts_3d_ok"]:
            ply_path = os.path.join(self.save_dir, f"pointcloud_{ts}.ply")
            def _do_all(r=result, pp=ply_path, ss=saved):
                n = save_ply(r["pts_3d_ok"], pp)
                ss.append(f"ply({n:,}pts)")
                self.after(0, lambda: self.v_status.set(
                    f"✓ 전체 저장 완료: {', '.join(ss)}  →  {self.save_dir}"))
            threading.Thread(target=_do_all, daemon=True).start()
            return

    self.v_status.set(f"✓ 저장 완료: {', '.join(saved)}  →  {self.save_dir}")

def _change_dir(self):
    d = filedialog.askdirectory(initialdir=self.save_dir)
    if d:
        self.save_dir = d
        self.v_status.set(f"저장 경로: {d}")

def _on_close(self):
    self._stop_cam()
    self.destroy()
```

# ════════════════════════════════════════════════════════════════════════════

if **name** == “**main**”:
if not PIL_AVAILABLE:
print(”[ERROR] Pillow 필요: pip install Pillow”)
else:
print(”=” * 55)
print(”  OAK-D Pro  IR Dot Depth Point Cloud Studio”)
print(f”  depthai : {‘✓’ if DEPTHAI_AVAILABLE else ‘✗  → DEMO 모드’}”)
print(f”  open3d  : {‘✓’ if OPEN3D_AVAILABLE else ‘✗  → 수동 PLY 작성’}”)
print(”=” * 55)
app = App()
app.geometry(“1380x900”)
app.minsize(1100, 750)
app.mainloop()