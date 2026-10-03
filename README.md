# jetson-realsense-yolo

RealSense D435i と YOLO(セグメンテーション + トラッキング)で人物を検出し、人物マスク内の Depth の中央値からカメラまでの距離を表示します。
処理区間ごとの時間を計測する機能と、TensorRT エンジンでの実行に対応しています。

- 前回の記事:[Jetson Orin Nano + RealSense + YOLOでリアルタイム人物検出を動かしてみた](https://qiita.com/tkmtmnt/items/edf50345684921c85823)
- 計測と高速化の記録:<!-- TODO: 続編記事の URL -->(Qiita)

## 動作確認環境

- Jetson Orin Nano Developer Kit Super(L4T R36.4.7)/ Python 3.10 / TensorRT 10.3
- PC(GeForce RTX 5080)/ Python 3.12
- ultralytics 8.4.165 / RealSense D435i

## セットアップ

```bash
# torch / torchvision は環境に合った CUDA 版を先にインストールする(Jetson は JetPack 対応版)
pip install -r requirements.txt
```

モデル `yolo26n-seg.pt` は、初回実行時に自動でダウンロードされます。

## 実行

```bash
python3 realsense_yolo_depth.py                 # 画面表示あり。q キーで終了
python3 realsense_yolo_depth.py --profile --no-display --max-frames 600   # 処理時間を計測
```

| 引数 | 内容 |
|---|---|
| `--profile` | 区間ごとの平均処理時間と FPS を表示する(最初の 30 フレームは除外) |
| `--no-display` / `--max-frames N` | 画面表示なし / N フレームで自動終了 |
| `--model PATH` | モデルを指定する(TensorRT の `.engine` も可) |
| `--tracker {botsort,bytetrack}` | トラッカーを選ぶ(既定:botsort) |
| `--fast-draw` | YOLO 側のマスク描画を省く(人物領域は後で塗りつぶすため、表示結果は同じ) |

## TensorRT(Jetson)

エンジンは実行するマシン上で作ります。`imgsz=480,640` を必ず指定してください(指定しないと 640×640 になり、人物マスクの位置がずれます)。

```bash
yolo export model=yolo26n-seg.pt format=engine half=True imgsz=480,640
python3 realsense_yolo_depth.py --model yolo26n-seg.engine
```

<!-- TODO: ライセンスを決めたら追記(Ultralytics は AGPL-3.0) -->
