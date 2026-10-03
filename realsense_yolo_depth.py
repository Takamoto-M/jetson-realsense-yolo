import argparse

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

from perf_profiler import FrameProfiler, NullProfiler, collect_env_info


# ==========================================
# コマンドライン引数(指定なしなら従来どおりの動作)
# ==========================================
parser = argparse.ArgumentParser(
    description="RealSense + YOLO 人物検出・距離計測"
)
parser.add_argument(
    "--profile", action="store_true",
    help="性能計測を有効にする(CSV・環境情報・統計を出力)"
)
parser.add_argument(
    "--warmup", type=int, default=30,
    help="統計から除外する起動直後のフレーム数(default: 30)"
)
parser.add_argument(
    "--report-interval", type=int, default=100,
    help="統計を表示するフレーム間隔(default: 100)"
)
parser.add_argument(
    "--profile-dir", default="profile_logs",
    help="計測結果の出力先ディレクトリ(default: profile_logs)"
)
parser.add_argument(
    "--no-display", action="store_true",
    help="画面表示を行わない"
)
parser.add_argument(
    "--max-frames", type=int, default=0,
    help="指定フレーム数を処理したら自動終了する(0: 無制限)"
)
parser.add_argument(
    "--model", default="yolo26n-seg.pt",
    help="YOLO モデルファイル(.pt / TensorRT の .engine など。default: yolo26n-seg.pt)"
)
args = parser.parse_args()


# ==========================================
# YOLO Segmentationモデル
# ==========================================
model = YOLO(args.model)


# ==========================================
# RealSense設定
# ==========================================
pipeline = rs.pipeline()
config = rs.config()

# RGB
config.enable_stream(
    rs.stream.color,
    640,
    480,
    rs.format.bgr8,
    30
)

# Depth
config.enable_stream(
    rs.stream.depth,
    640,
    480,
    rs.format.z16,
    30
)

profile = pipeline.start(config)


# ==========================================
# 性能計測
# ==========================================
if args.profile:
    prof = FrameProfiler(
        args.profile_dir,
        warmup=args.warmup,
        report_interval=args.report_interval,
        target_fps=30,
    )
    prof.save_env_info(
        collect_env_info(
            profile,
            extra={
                "model": args.model,
                "stream": "color 640x480 bgr8 30fps / depth 640x480 z16 30fps",
                "args": vars(args),
            },
        )
    )
else:
    prof = NullProfiler()

processed_frames = 0
mask_shape_warned = False


# ==========================================
# Depth → Color の位置合わせ
# ==========================================
align = rs.align(rs.stream.color)


try:
    while True:

        prof.begin_frame()

        # ----------------------------------
        # RealSenseからフレーム取得
        # ----------------------------------
        frames = pipeline.wait_for_frames()
        prof.lap("capture_wait")

        # Depth画像をRGB画像の座標系へ合わせる
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(
            color_frame.get_data()
        )

        depth_image = np.asanyarray(
            depth_frame.get_data()
        )
        prof.lap("capture_align")


        # ----------------------------------
        # YOLO Segmentation + Tracking
        # ----------------------------------
        results = model.track(
            source=color_image,
            classes=[0],       # personのみ
            device=0,          # RTX 5080
            persist=True,
            verbose=False
        )
        prof.sync()
        prof.lap("yolo")

        result = results[0]

        # YOLOによる描画
        annotated_frame = result.plot()
        prof.lap("draw")


        # ----------------------------------
        # Segmentation Maskが存在する場合
        # ----------------------------------
        if result.masks is not None:

            masks = result.masks.data.cpu().numpy()

            # マスクと画像のサイズが違うと、下の単純リサイズで位置がずれる
            # (例: 640x640 固定で作った TensorRT エンジン)。一度だけ警告する
            if masks.shape[1:] != depth_image.shape and not mask_shape_warned:
                print(
                    f"[WARNING] マスクサイズ {masks.shape[1:]} が画像サイズ "
                    f"{depth_image.shape} と異なります。人物領域と距離が"
                    "ずれる可能性があります(エンジンは imgsz=480,640 で"
                    "作成してください)"
                )
                mask_shape_warned = True

            # Track ID
            if (
                result.boxes is not None
                and result.boxes.id is not None
            ):
                track_ids = (
                    result.boxes.id
                    .int()
                    .cpu()
                    .tolist()
                )
            else:
                track_ids = [None] * len(masks)
            prof.lap("mask")


            # ----------------------------------
            # 人物ごとに処理
            # ----------------------------------
            for mask, track_id in zip(masks, track_ids):

                # YOLO MaskをDepth画像サイズへ合わせる
                mask_resized = cv2.resize(
                    mask,
                    (
                        depth_image.shape[1],
                        depth_image.shape[0]
                    ),
                    interpolation=cv2.INTER_NEAREST
                )

                person_mask = mask_resized > 0.5
                prof.lap("mask")

                # 人物領域を濃い青で塗る
                annotated_frame[person_mask] = (155, 0, 0)
                prof.lap("draw")


                # ----------------------------------
                # 人物Mask内のDepthを取得
                # ----------------------------------
                person_depth = depth_image[person_mask]

                # 無効Depth（0）を除外
                valid_depth = person_depth[
                    person_depth > 0
                ]

                if len(valid_depth) == 0:
                    prof.lap("depth")
                    continue


                # ----------------------------------
                # Depth scale取得
                # ----------------------------------
                depth_scale = (
                    profile
                    .get_device()
                    .first_depth_sensor()
                    .get_depth_scale()
                )


                # ----------------------------------
                # 距離の中央値
                # ----------------------------------
                distance_m = (
                    np.median(valid_depth)
                    * depth_scale
                )
                prof.lap("depth")


                # ----------------------------------
                # 表示位置を取得
                # ----------------------------------
                ys, xs = np.where(person_mask)

                if len(xs) == 0:
                    prof.lap("draw")
                    continue

                x = int(np.mean(xs))
                y = int(np.mean(ys))


                # ----------------------------------
                # 距離表示
                # ----------------------------------
                text_label = "カメラからの距離："
                text_distance = f"{distance_m:.2f} m"

                cv2.putText(
                    annotated_frame,
                    text_label,
                    (x - 60, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                cv2.putText(
                    annotated_frame,
                    text_distance,
                    (x - 60, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
                prof.lap("draw")


        # ----------------------------------
        # 表示
        # ----------------------------------
        if not args.no_display:
            cv2.imshow(
                "RealSense + YOLO + Depth",
                annotated_frame
            )

            # qで終了
            key = cv2.waitKey(1) & 0xFF
            prof.lap("display")
            if key == ord("q"):
                break

        prof.end_frame(
            n_persons=(
                len(result.boxes) if result.boxes is not None else 0
            ),
            rs_frame_number=color_frame.get_frame_number(),
            yolo_speed=result.speed,
        )

        # 指定フレーム数で自動終了
        processed_frames += 1
        if args.max_frames > 0 and processed_frames >= args.max_frames:
            break


finally:

    prof.close()
    pipeline.stop()
    if not args.no_display:
        cv2.destroyAllWindows()
