# jetson-realsense-yolo

RealSense D435i と YOLO を使い、人物検出・追跡と距離計測を行うプログラムです。
検出した人物領域を青く表示し、その領域内の有効な Depth 値の中央値から、カメラまでの距離を求めます。また、処理区間ごとの時間を計測する機能と、TensorRT エンジンでの実行に対応しています。

Jetson Orin Nano でのリアルタイム動作を目指し、処理区間ごとの時間を計測して改善しました。TensorRT の導入とトラッカーの変更により、処理時間の合計を約 61 ms から約 24 ms に短縮し、実測 FPS は約 16 から約 30 に向上しました。

- 前回の記事：[Jetson Orin Nano + RealSense + YOLOでリアルタイム人物検出を動かしてみた](https://qiita.com/tkmtmnt/items/edf50345684921c85823)
- 今回の記事：[Claude Code で Jetson Orin Nano 上の YOLO を計測・高速化してみた(16 FPS → 30 FPS)](https://qiita.com/tkmtmnt/items/320f8e6ee7b3d09273da)
- 計測と改善の詳細：[docs/profiling.md](docs/profiling.md)（計測ログは [results/](results/)）
- AI エージェントを使った進め方：[docs/claude-code-workflow.md](docs/claude-code-workflow.md)

## 動作確認環境

- Jetson Orin Nano Developer Kit Super（L4T R36.4.7）/ Python 3.10 / TensorRT 10.3
- PC（GeForce RTX 5080）/ Python 3.12
- ultralytics 8.4.165 / RealSense D435i

## セットアップ

```bash
# torch / torchvision は環境に合った CUDA 版を先にインストールする（Jetson は JetPack 対応版）
pip install -r requirements.txt
```

モデル `yolo26n-seg.pt` は、初回実行時に自動でダウンロードされます。

## 実行

```bash
# 画面表示あり。q キーで終了
python3 realsense_yolo_depth.py

# 画面表示なしで処理時間を計測
python3 realsense_yolo_depth.py --profile --no-display --max-frames 600
```

| 引数 | 内容 |
|---|---|
| `--profile` | 区間ごとの処理時間と FPS を表示する（最初の 30 フレームは集計から除外） |
| `--no-display` / `--max-frames N` | 画面表示なし / N フレームで自動終了 |
| `--model PATH` | モデルを指定する（TensorRT の `.engine` も使用可能） |
| `--tracker {botsort,bytetrack}` | トラッカーを選ぶ（既定：botsort） |

## TensorRT（Jetson）

TensorRT エンジンは、実行する Jetson 上で作成します。
このプログラムでは 640×480 の画像を使用するため、変換時に `imgsz=480,640` を指定します。

今回の環境では、この指定を省くと入力が 640×640 になり、人物マスクの位置がずれる現象がありました。距離計測に使う領域にも影響するため、変換後はマスクが人物に重なっていることを確認してください。

```bash
yolo export model=yolo26n-seg.pt format=engine half=True imgsz=480,640

# 処理時間を短縮した設定で実行
python3 realsense_yolo_depth.py --model yolo26n-seg.engine --tracker bytetrack
```

## 開発について

コードの作成、Jetson 上での実行テスト・ログ集計には Claude Code を使用しました。開発方針の決定、コードの最終確認、AI エージェントによる作業結果の確認は、自身で行っています。

作業の進め方は [docs/claude-code-workflow.md](docs/claude-code-workflow.md) にまとめています。