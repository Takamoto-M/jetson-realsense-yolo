# jetson-realsense-yolo

Intel RealSense D435i と YOLO(セグメンテーション + トラッキング)で人物を検出し、人物マスク内の Depth の中央値からカメラまでの距離を表示するプログラムです。
各処理の時間を計測する機能と、TensorRT エンジンでの実行に対応しています。

- 前回の記事:[Jetson Orin Nano + RealSense + YOLOでリアルタイム人物検出を動かしてみた](https://qiita.com/tkmtmnt/items/edf50345684921c85823)
<!-- TODO: 続編記事の URL を追記 -->

## ファイル構成

| ファイル | 内容 |
|---|---|
| `realsense_yolo_depth.py` | 本体(RealSense 取得 → YOLO → 人物マスク → 距離計算 → 表示) |
| `perf_profiler.py` | 処理時間の計測モジュール(`--profile` 指定時のみ使用) |

## 動作確認環境

| | PC | Jetson |
|---|---|---|
| ハードウェア | Ryzen 9 9900X / GeForce RTX 5080 | Jetson Orin Nano Developer Kit Super(L4T R36.4.7) |
| Python | 3.12.3 | 3.10.12 |
| torch | 2.14.0+cu130 | 2.10.0(CUDA 12.6) |
| ultralytics | 8.4.165 | 8.4.165 |
| pyrealsense2 | 2.58.4 | 2.58.4 |
| TensorRT | — | 10.3.0(JetPack 付属) |
| カメラ | RealSense D435i(FW 5.16.0.1) | 同左 |

## セットアップ

```bash
# torch / torchvision は環境に合った CUDA 版を先にインストールする
# (Jetson では JetPack に対応したものが必要。PyPI の通常版では GPU が使えない)
pip install -r requirements.txt
```

モデル `yolo26n-seg.pt` は、プログラムと同じフォルダにない場合、初回実行時に Ultralytics が自動でダウンロードします。

## 実行方法

モデルファイルは相対パスで読み込むので、このフォルダで実行してください。

```bash
python3 realsense_yolo_depth.py          # 画面表示あり。q キーで終了
```

| 引数 | 内容 |
|---|---|
| `--model PATH` | 使うモデル(既定:`yolo26n-seg.pt`。TensorRT の `.engine` も指定可) |
| `--no-display` | 画面表示をしない |
| `--max-frames N` | N フレーム処理したら自動で終了する(0 は無制限) |
| `--profile` | 処理時間を計測する |
| `--warmup N` | 統計から除外する起動直後のフレーム数(既定:30) |
| `--report-interval N` | 統計を表示する間隔(既定:100 フレーム) |
| `--profile-dir DIR` | 計測結果の出力先(既定:`profile_logs`) |

## 処理時間の計測

```bash
python3 realsense_yolo_depth.py --profile --no-display --max-frames 600
```

一定フレームごとに、区間ごとの平均・p50・p95・最大値と FPS を表示します。終了時には `--profile-dir` に次の 3 ファイルを保存します。

- `profile_*.csv`:フレームごとの生データ(ウォームアップ分も含め、集計なし)
- `profile_*_env.json`:実行環境(CPU、GPU、CUDA、主要ライブラリのバージョン、カメラ情報)
- `profile_*_summary.txt`:終了時の統計

主な区間は次のとおりです。

| 区間 | 内容 |
|---|---|
| `capture_wait` | `wait_for_frames()`。カメラの次のフレームを待つ時間を含む |
| `capture` | `capture_wait` + Depth の位置合わせ |
| `yolo` | `model.track()`(GPU の完了待ちを含む) |
| `mask` / `depth` | マスク処理 / マスク内 Depth の中央値計算 |
| `draw` / `display` | 描画 / `imshow` + `waitKey` |
| `busy` | 全体からカメラ待ちを除いた時間(=処理能力の目安) |

実測 FPS はカメラの 30 FPS で頭打ちになるので、処理能力は `busy` と「処理のみ換算 FPS」で比べてください。

## TensorRT エンジンで実行する(Jetson)

エンジンは実行するマシン上で作ります。**`imgsz=480,640` は必ず指定してください。** 指定しないと 640×640 固定のエンジンになり、人物マスクの位置がずれます。その場合、実行時に `[WARNING]` が表示されます。

```bash
sudo jetson_clocks   # 計測時はクロックを固定(再起動で元に戻る)
yolo export model=yolo26n-seg.pt format=engine half=True imgsz=480,640
python3 realsense_yolo_depth.py --model yolo26n-seg.engine
```

エクスポートには `onnx` と `onnxslim` が必要です。また、venv を使っている場合は、JetPack の `tensorrt` が venv から import できる必要があります。

## 計測結果の例

人物 1 人、画面表示なし、600 フレーム(最初の 30 フレームを除く)の平均です。Jetson は電源モード MAXN_SUPER + `jetson_clocks` で計測しました。

| 環境 | YOLO 推論 | YOLO 全体 | busy | 実測 FPS | 取りこぼし |
|---|---|---|---|---|---|
| PC(`.pt`) | 3.3 ms | 6.8 ms | 11.4 ms | 30.0 | 0 |
| Jetson(`.pt`) | 30.9 ms | 50.1 ms | 60.1 ms | 16.6 | 460 |
| Jetson(TensorRT FP16) | 3.8 ms | 22.1 ms | 30.2 ms | 29.9 | 1 |

YOLO 推論は Ultralytics が計測した値、YOLO 全体は前処理・後処理・トラッキングを含む `model.track()` の時間です。

## 既知の制限

- 距離ラベルの日本語(「カメラからの距離」)は、OpenCV の `putText` が対応していないため正しく表示されません。
- 距離は、人物マスク内の有効な Depth(0 以外)の中央値です。

<!-- TODO: ライセンスを決めたら追記(Ultralytics は AGPL-3.0) -->
