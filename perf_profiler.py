"""
処理区間ごとの時間を計測して、コマンドラインに表示する簡易プロファイラ

lap(name) で「前回の lap からの経過時間」を区間 name に加算し、
report_interval フレームごとに各区間の平均時間と FPS を表示する。
起動直後の warmup フレームは統計から除外する。
"""

import time

# 表示順。yolo は Ultralytics の計測値(result.speed)を使って
# 推論 / 前処理+後処理 / それ以外(トラッキング等)に分けて表示する
ROWS = (
    ("wait", "カメラ待ち"),
    ("camera", "カメラ取得・位置合わせ"),
    ("yolo_infer", "YOLO 推論"),
    ("yolo_prepost", "YOLO 前処理+後処理"),
    ("yolo_track", "YOLO トラッキング等"),
    ("distance", "マスク処理+距離計算"),
    ("draw", "描画"),
    ("display", "画面表示"),
    ("busy", "処理合計(カメラ待ちを除く)"),
    ("total", "フレーム全体"),
)


class SectionTimer:

    def __init__(self, enabled, warmup=30, report_interval=100):
        self.enabled = enabled
        self.warmup = warmup
        self.report_interval = report_interval
        self.frame = 0
        self._sum = {}       # report_interval 内の合計
        self._sum_all = {}   # ウォームアップ後の全体の合計
        self._n = 0
        self._n_all = 0

        # GPU は非同期に動くので、YOLO 区間の終わりで完了を待つ
        self._cuda = None
        if enabled:
            import torch

            if torch.cuda.is_available():
                self._cuda = torch.cuda

    def start_frame(self):
        if not self.enabled:
            return
        self._t0 = self._last = time.perf_counter()
        self._cur = dict.fromkeys(
            ("wait", "camera", "yolo", "distance", "draw", "display"), 0.0
        )

    def lap(self, name):
        """前回の lap(または start_frame)からの経過時間を name に加算する"""
        if not self.enabled:
            return
        now = time.perf_counter()
        self._cur[name] += (now - self._last) * 1000
        self._last = now

    def sync(self):
        if self._cuda is not None:
            self._cuda.synchronize()

    def end_frame(self, yolo_speed):
        if not self.enabled:
            return
        now = time.perf_counter()
        self.frame += 1

        c = self._cur
        infer = yolo_speed.get("inference") or 0.0
        prepost = (yolo_speed.get("preprocess") or 0.0) + (
            yolo_speed.get("postprocess") or 0.0
        )
        total = (now - self._t0) * 1000
        values = {
            "wait": c["wait"],
            "camera": c["camera"],
            "yolo_infer": infer,
            "yolo_prepost": prepost,
            "yolo_track": c["yolo"] - infer - prepost,
            "distance": c["distance"],
            "draw": c["draw"],
            "display": c["display"],
            "busy": total - c["wait"],
            "total": total,
        }

        if self.frame <= self.warmup:
            # 最後のウォームアップフレームの終了時刻から計測を始める
            self._t_start = self._t_interval = now
            return
        if self._n_all == 0 and self.warmup == 0:
            self._t_start = self._t_interval = self._t0

        for k, v in values.items():
            self._sum[k] = self._sum.get(k, 0.0) + v
            self._sum_all[k] = self._sum_all.get(k, 0.0) + v
        self._n += 1
        self._n_all += 1
        self._t_end = now

        if self.report_interval > 0 and self._n >= self.report_interval:
            self._print(
                f"frames {self.frame - self._n + 1}-{self.frame}",
                self._sum, self._n, now - self._t_interval,
            )
            self._sum, self._n, self._t_interval = {}, 0, now

    def summary(self):
        if not self.enabled:
            return
        if self._n_all == 0:
            print(f"[profile] 計測対象のフレームがありません(warmup {self.warmup})")
            return
        self._print(
            f"SUMMARY {self._n_all} frames(最初の {self.warmup} フレームを除く)",
            self._sum_all, self._n_all, self._t_end - self._t_start,
        )

    @staticmethod
    def _print(title, sums, n, elapsed_s):
        fps = n / elapsed_s if elapsed_s > 0 else float("nan")
        print(f"[profile] {title}  FPS {fps:.1f}")
        for key, label in ROWS:
            print(f"  {sums[key] / n:7.2f} ms  {label}")
