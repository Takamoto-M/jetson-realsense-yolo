# 処理時間の計測と高速化の記録

Jetson Orin Nano 上で `realsense_yolo_depth.py` の処理時間を区間ごとに計測し、ボトルネックを順に改善した記録です。
計測ログは [`results/jetson_20261003/`](../results/jetson_20261003/) にあります。

## 計測条件

| 項目 | 内容 |
|---|---|
| ボード | Jetson Orin Nano Developer Kit Super(L4T R36.4.7)、電源モード MAXN_SUPER |
| カメラ | RealSense D435i(color / depth とも 640×480、30 FPS) |
| モデル | yolo26n-seg(Ultralytics 8.4.165)、TensorRT 10.3.0 |
| シーン | カメラの前に 1 人が立った状態 |
| 計測 | 画面表示なし、600 フレーム。最初の 30 フレームを除いた 570 フレームの平均 |
| コマンド | `python3 realsense_yolo_depth.py --profile --no-display --max-frames 600 [オプション]` |

## 計測方法

画像1枚を処理する流れを以下の区間に分け、それぞれにかかった時間を計測します。計測には、`perf_profiler.py` を使用しています。

| 区間 | 内容 |
|---|---|
| カメラ待ち | `pipeline.wait_for_frames()` |
| カメラ取得・位置合わせ | `align.process()` から numpy 配列化まで |
| YOLO 推論 / 前処理+後処理 | Ultralytics の `result.speed` の値 |
| YOLO トラッキング等 | `model.track()` 全体から、上の 2 つを引いた残り |
| マスク処理+距離計算 | マスクの CPU 転送・リサイズ、マスク内 Depth の中央値計算 |
| 描画 | `result.plot()`、人物領域の塗りつぶし、距離テキストの描画 |
| 画面表示 | `cv2.imshow()` + `cv2.waitKey()` |
| 処理合計 | フレーム全体からカメラ待ちを除いた時間 |

- GPU は非同期に動くので、YOLO の区間の終わりで `torch.cuda.synchronize()` を呼び、GPU処理が終わるまでCPUを待たせています。
- 実測 FPS はカメラの 30 FPS で頭打ちで最大になります。

## 結果

| 区間(m秒) | 計測① | 計測② | 計測③ |
|---|---|---|---|
| 概要 | PyTorchでの推論 | TensorRTでの推論 | 計測② + YOLOのCPU処理改善 |
| カメラ取得・位置合わせ | 1.34 | 1.18 | 0.93 |
| YOLO 推論 | 30.89 | **3.73** | 3.74 |
| YOLO 前処理+後処理 | 5.97 | 5.96 | 6.10 |
| YOLO トラッキング等 | 13.14 | 13.20 | **3.23** |
| マスク処理+距離計算 | 1.74 | 2.19 | 1.81 |
| 描画 | 7.95 | 10.02 | 8.49 |
| **処理合計** | **61.06** | **36.30** | **24.31** |
| 実測 FPS | 16.4 | 27.3 | 30.0 |
| ログ | `1_pt.txt` | `2_engine.txt` | `3_bytetrack.txt` |

| 設定 | 実行コマンド |
|---|---|
| ① | `python3 realsense_yolo_depth.py --profile --no-display --max-frames 600` |
| ② | `python3 realsense_yolo_depth.py --profile --no-display --max-frames 600 --model yolo26n-seg.engine` |
| ③ | `python3 realsense_yolo_depth.py --profile --no-display --max-frames 600 --model yolo26n-seg.engine --tracker bytetrack` |

## 各改善内容の詳細

### ① → ② TensorRT 化

PyTorch形式（.pt）のYOLOモデルを、JetsonのGPUで高速に動作するTensorRT形式（.engine）に変換しました。

```bash
# Jetson上で実行
yolo export model=yolo26n-seg.pt format=engine half=True imgsz=480,640
```

これにより、YOLOの推論時間は 30.89 ms → 3.73 ms（約8.3倍高速化） しました。

注意点は imgsz=480,640 の指定です。指定しないと640×640のエンジンが作られ、今回使用する640×480のカメラ画像と縦横比が合わず、人物マスクの位置がずれてしまいました。

そのため、カメラ画像に合わせて imgsz=480,640 を指定しています。

なお、Jetson で変換するには、事前に次の準備が必要でした。

- venv から JetPack の `tensorrt`(システムの Python 側にある)を import できるようにする(今回は venv の `site-packages` にシンボリックリンクを作成)
- `pip install onnx onnxslim` で、変換に使うパッケージを入れる

### ② → ③ トラッカーを ByteTrack に変更

最初に使用していたBoT-SORTには、カメラ自体が動いた場合に映像の動きを補正する処理があります。

今回はRealSenseを固定して使用するため、この処理は必要ありません。そこで、よりシンプルな ByteTrack に変更しました。

これにより、トラッキング等の処理時間は 13.20 ms → 3.23 ms まで短縮できました。

なお、車載カメラなどカメラ自体が動く場合には、BoT-SORTの補正が有効になるとのことです。

## 参考:電源モードの影響

改善の前に、電源モードを 25W から MAXN_SUPER に変えたところ、処理合計(PyTorch)は 92.6 ms から 60.1 ms になりました。この 2 つの値は、計測コードを整理する前の版で計測したものです。

## 補足:失敗したログについて

`results/` にある `3_bytetrack_failed.txt` は、TensorRT エンジンの読み込み中にエラー(`cudaErrorStreamCaptureInvalidated`)で終了した実行のログです。同じ設定で再実行すると正常に動きました(`3_bytetrack.txt`)。現時点で、原因については調べていません。
