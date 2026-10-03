"""
realsense_yolo_depth.py 用の簡易性能計測モジュール

- 各フレームの区間処理時間を time.perf_counter() で計測する
- フレームごとの生データを CSV に保存する(集計しない)
- 一定フレーム数ごとに区間統計と FPS を表示する
- 実行開始時に環境情報を JSON に保存する

計測 OFF 時は NullProfiler を使い、何もしない(GPU 同期も行わない)。
"""

import csv
import datetime
import importlib.metadata
import json
import os
import platform
import sys
import time

import numpy as np


# 区間名(CSV の列順・表示順)
# capture_wait  : pipeline.wait_for_frames()(カメラ待ち時間を含む)
# capture_align : align.process() ~ numpy 配列化
# yolo          : model.track()(前処理・推論・後処理・トラッキング。GPU 同期込み)
# mask          : マスクの CPU 転送・リサイズ・2値化
# depth         : マスク内 Depth 抽出・depth scale 取得・中央値計算
# draw          : result.plot()・人物領域塗り・表示位置計算・putText
# display       : cv2.imshow() + cv2.waitKey()(画面表示ありの場合のみ)
SECTIONS = (
    "capture_wait",
    "capture_align",
    "yolo",
    "mask",
    "depth",
    "draw",
    "display",
)

# 統計表示する系列(capture は wait + align の合計、busy は total - wait)
STAT_ROWS = (
    "capture",
    "capture_wait",
    "yolo",
    "mask",
    "depth",
    "draw",
    "display",
    "total",
    "busy",
)

CSV_COLUMNS = (
    ["frame", "t_start_s", "warmup"]
    + [f"{s}_ms" for s in SECTIONS]
    + [
        "capture_ms",
        "other_ms",
        "total_ms",
        "busy_ms",
        "yolo_pre_ms",
        "yolo_inf_ms",
        "yolo_post_ms",
        "n_persons",
        "rs_frame_number",
    ]
)


class NullProfiler:
    """計測 OFF 時に使う何もしないプロファイラ"""

    def begin_frame(self):
        pass

    def lap(self, name):
        pass

    def sync(self):
        pass

    def end_frame(self, **kwargs):
        pass

    def close(self):
        pass


class FrameProfiler:

    def __init__(
        self,
        out_dir,
        warmup=30,
        report_interval=100,
        target_fps=30,
        sync_cuda=True,
    ):
        self.warmup = warmup
        self.report_interval = report_interval
        self.deadline_ms = 1000.0 / target_fps

        # GPU 同期(CUDA が使える場合のみ)
        self._torch = None
        if sync_cuda:
            import torch

            if torch.cuda.is_available():
                self._torch = torch

        os.makedirs(out_dir, exist_ok=True)
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(out_dir, f"profile_{run_id}.csv")
        self.env_path = os.path.join(out_dir, f"profile_{run_id}_env.json")
        self.summary_path = os.path.join(
            out_dir, f"profile_{run_id}_summary.txt"
        )

        self._csv_file = open(self.csv_path, "w", newline="")
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow(CSV_COLUMNS)

        self._t_origin = time.perf_counter()
        self._frame = 0
        self._t0 = None
        self._last = None
        self._acc = {}

        # 統計用(ウォームアップ除外後)
        self._all = {k: [] for k in STAT_ROWS}
        self._interval = {k: [] for k in STAT_ROWS}
        self._measure_start = None      # 統計対象区間の開始時刻
        self._interval_start = None
        self._last_end = None
        self._closed = False

        # RealSense フレーム番号の飛び(取りこぼし)の集計
        self._last_rs_frame = None
        self._dropped_all = 0
        self._dropped_interval = 0

    # ----------------------------------
    # 環境情報
    # ----------------------------------
    def save_env_info(self, info):
        with open(self.env_path, "w") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        print("[profile] 環境情報:")
        for key in (
            "cpu_model",
            "gpu_name",
            "cuda_available",
            "torch_cuda_version",
            "python_version",
        ):
            print(f"  {key:20s}: {info.get(key)}")
        for name, ver in info.get("packages", {}).items():
            print(f"  {name:20s}: {ver}")
        print(f"[profile] CSV      : {self.csv_path}")
        print(f"[profile] 環境情報 : {self.env_path}")

    # ----------------------------------
    # フレーム計測
    # ----------------------------------
    def begin_frame(self):
        now = time.perf_counter()
        self._t0 = now
        self._last = now
        self._acc = dict.fromkeys(SECTIONS, 0.0)

    def lap(self, name):
        """前回の lap(または begin_frame)からの経過時間を name に加算する"""
        now = time.perf_counter()
        self._acc[name] += now - self._last
        self._last = now

    def sync(self):
        """GPU の非同期処理の完了を待つ(CUDA が無い場合は何もしない)"""
        if self._torch is not None:
            self._torch.cuda.synchronize()

    def end_frame(self, n_persons=0, rs_frame_number=None, yolo_speed=None):
        now = time.perf_counter()
        self._frame += 1

        ms = {k: v * 1e3 for k, v in self._acc.items()}
        total = (now - self._t0) * 1e3
        capture = ms["capture_wait"] + ms["capture_align"]
        other = total - sum(ms.values())
        busy = total - ms["capture_wait"]

        speed = yolo_speed or {}
        is_warmup = self._frame <= self.warmup

        self._csv.writerow(
            [self._frame, f"{self._t0 - self._t_origin:.6f}", int(is_warmup)]
            + [f"{ms[s]:.4f}" for s in SECTIONS]
            + [
                f"{capture:.4f}",
                f"{other:.4f}",
                f"{total:.4f}",
                f"{busy:.4f}",
                _fmt(speed.get("preprocess")),
                _fmt(speed.get("inference")),
                _fmt(speed.get("postprocess")),
                n_persons,
                "" if rs_frame_number is None else rs_frame_number,
            ]
        )

        # 前フレームからの RealSense フレーム番号の飛び(ウォームアップ中も追跡)
        dropped = 0
        if rs_frame_number is not None:
            if self._last_rs_frame is not None:
                dropped = max(0, rs_frame_number - self._last_rs_frame - 1)
            self._last_rs_frame = rs_frame_number

        if is_warmup:
            # 最後のウォームアップフレームの終了時刻から統計を開始
            self._measure_start = now
            self._interval_start = now
            if self._frame == self.warmup:
                print(
                    f"[profile] ウォームアップ {self.warmup} フレーム完了。"
                    "統計を開始します"
                )
            return

        if self._measure_start is None:
            # warmup=0 の場合
            self._measure_start = self._t0
            self._interval_start = self._t0
        self._last_end = now
        self._dropped_all += dropped
        self._dropped_interval += dropped

        values = dict(ms)
        values["capture"] = capture
        values["total"] = total
        values["busy"] = busy
        for k in STAT_ROWS:
            self._all[k].append(values[k])
            self._interval[k].append(values[k])

        n = len(self._interval["total"])
        if self.report_interval > 0 and n >= self.report_interval:
            first = self._frame - n + 1
            title = f"frames {first}-{self._frame} ({n} frames)"
            print(
                self._format_stats(
                    title,
                    self._interval,
                    now - self._interval_start,
                    self._dropped_interval,
                    with_p99=False,
                )
            )
            self._csv_file.flush()
            self._interval = {k: [] for k in STAT_ROWS}
            self._dropped_interval = 0
            self._interval_start = now

    # ----------------------------------
    # 統計表示
    # ----------------------------------
    def _format_stats(self, title, data, elapsed_s, dropped, with_p99):
        n = len(data["total"])
        busy = np.asarray(data["busy"])

        fps = n / elapsed_s if elapsed_s > 0 else float("nan")
        busy_fps = 1000.0 / busy.mean()
        # wait_for_frames() はカメラ周期に合わせて待つため total は常に
        # 約 33ms になる。処理が周期に間に合っているかは busy で判定する
        miss = (busy > self.deadline_ms).mean() * 100.0

        cols = ["mean", "p50", "p95"]
        if with_p99:
            cols.append("p99")
        cols.append("max")

        lines = [
            f"[profile] {title}",
            f"  FPS(実測) {fps:6.2f}   "
            f"処理のみ換算FPS(1000/mean busy) {busy_fps:6.2f}   "
            f"busy>{self.deadline_ms:.1f}ms {miss:5.1f}%   "
            f"カメラフレーム取りこぼし {dropped}",
            "  " + f"{'section(ms)':14s}" + "".join(f"{c:>9s}" for c in cols),
        ]
        for k in STAT_ROWS:
            a = np.asarray(data[k])
            vals = [a.mean(), np.percentile(a, 50), np.percentile(a, 95)]
            if with_p99:
                vals.append(np.percentile(a, 99))
            vals.append(a.max())
            lines.append(
                "  " + f"{k:14s}" + "".join(f"{v:9.2f}" for v in vals)
            )
        return "\n".join(lines)

    def close(self):
        if self._closed:
            return
        self._closed = True

        self._csv_file.close()

        n = len(self._all["total"])
        if n == 0:
            msg = (
                f"[profile] 統計対象フレームがありません"
                f"(処理フレーム {self._frame}、ウォームアップ {self.warmup})"
            )
        else:
            first = self._frame - n + 1
            msg = self._format_stats(
                f"SUMMARY frames {first}-{self._frame} ({n} frames, "
                f"warmup {self.warmup} excluded)",
                self._all,
                self._last_end - self._measure_start,
                self._dropped_all,
                with_p99=True,
            )
        print(msg)

        with open(self.summary_path, "w") as f:
            f.write(msg + "\n")
        print(f"[profile] CSV      : {self.csv_path}")
        print(f"[profile] サマリ   : {self.summary_path}")


def _fmt(v):
    return "" if v is None else f"{v:.4f}"


# ==========================================
# 環境情報の収集
# ==========================================
def _read_cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _read_file(path):
    try:
        with open(path) as f:
            return f.read().strip().strip("\x00")
    except OSError:
        return None


def _mem_total_gb():
    text = _read_file("/proc/meminfo")
    if not text:
        return None
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            return round(int(line.split()[1]) / 1024 / 1024, 2)
    return None


def collect_env_info(rs_profile=None, extra=None):
    info = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": _read_cpu_model(),
        "cpu_count": os.cpu_count(),
        "mem_total_gb": _mem_total_gb(),
        # Jetson の場合のみ取得できる
        "device_tree_model": _read_file("/proc/device-tree/model"),
        "nv_tegra_release": _read_file("/etc/nv_tegra_release"),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
    }

    packages = {}
    for name in (
        "numpy",
        "opencv-python",
        "torch",
        "torchvision",
        "ultralytics",
        "pyrealsense2",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    info["packages"] = packages

    try:
        import cv2

        info["cv2_version"] = cv2.__version__
    except ImportError:
        pass

    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        info["torch_cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            info["gpu_capability"] = f"{cap[0]}.{cap[1]}"
            info["gpu_count"] = torch.cuda.device_count()
        else:
            info["gpu_name"] = None
    except ImportError:
        info["cuda_available"] = None

    if rs_profile is not None:
        try:
            import pyrealsense2 as rs

            dev = rs_profile.get_device()
            info["realsense"] = {
                "name": dev.get_info(rs.camera_info.name),
                "serial": dev.get_info(rs.camera_info.serial_number),
                "firmware": dev.get_info(rs.camera_info.firmware_version),
                "usb_type": (
                    dev.get_info(rs.camera_info.usb_type_descriptor)
                    if dev.supports(rs.camera_info.usb_type_descriptor)
                    else None
                ),
            }
        except Exception as e:  # 環境情報の取得失敗で計測を止めない
            info["realsense"] = {"error": repr(e)}

    if extra:
        info.update(extra)
    return info
