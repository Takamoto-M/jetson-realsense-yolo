# 処理時間の計測と高速化の記録

Jetson Orin Nano 上で `realsense_yolo_depth.py` の処理時間を区間ごとに計測し、ボトルネックを順に改善した記録です。
生ログは [`results/jetson_20261003/`](../results/jetson_20261003/) にあります。

## 計測条件

| 項目 | 内容 |
|---|---|
| ボード | Jetson Orin Nano Developer Kit Super(L4T R36.4.7)、電源モード MAXN_SUPER + `sudo jetson_clocks` |
| カメラ | RealSense D435i(color / depth とも 640×480、30 FPS) |
| モデル | yolo26n-seg(Ultralytics 8.4.165)、TensorRT 10.3.0 |
| シーン | カメラの前に 1 人が立った状態 |
| 計測 | 画面表示なし、600 フレーム。最初の 30 フレームを除いた 570 フレームの平均 |
| コマンド | `python3 realsense_yolo_depth.py --profile --no-display --max-frames 600 [オプション]` |

## 計測方法

`perf_profiler.py` の `SectionTimer` で、`time.perf_counter()` を使い、「前回の区切りからの経過時間」を区間ごとに加算します。

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

- GPU は非同期に動くので、YOLO の区間の終わりで `torch.cuda.synchronize()` を呼び、推論の時間が後の区間に紛れ込まないようにしています。
- 実測 FPS はカメラの 30 FPS で頭打ちになります。処理能力は「処理合計」で比べます。30 FPS を保つには、33.3 ms 以内に収める必要があります。

## 結果

| 区間(ms) | ① PyTorch | ② TensorRT | ③ + ByteTrack | ④ + 描画の改善 |
|---|---|---|---|---|
| カメラ取得・位置合わせ | 1.34 | 1.18 | 0.93 | 0.91 |
| YOLO 推論 | 30.89 | **3.73** | 3.74 | 3.73 |
| YOLO 前処理+後処理 | 5.97 | 5.96 | 6.10 | 6.02 |
| YOLO トラッキング等 | 13.14 | 13.20 | **3.23** | 3.06 |
| マスク処理+距離計算 | 1.74 | 2.19 | 1.81 | 1.60 |
| 描画 | 7.95 | 10.02 | 8.49 | **5.53** |
| **処理合計** | **61.06** | **36.30** | **24.31** | **20.87** |
| 実測 FPS | 16.4 | 27.3 | 30.0 | 30.0 |
| ログ | `1_pt.txt` | `2_engine.txt` | `3_bytetrack.txt` | `4_fastdraw.txt` |

| 設定 | オプション |
|---|---|
| ① | なし |
| ② | `--model yolo26n-seg.engine` |
| ③ | ② + `--tracker bytetrack` |
| ④ | ③ + `--fast-draw` |

画面表示あり(④の設定、`5_display.txt`):画面表示 3.19 ms、処理合計 24.68 ms、実測 30.0 FPS。

## 各改善の内容

### ① → ② TensorRT 化

```bash
yolo export model=yolo26n-seg.pt format=engine half=True imgsz=480,640
```

推論は 30.89 ms から 3.73 ms になりました(約 8.3 倍)。

**`imgsz=480,640` の指定は必須です。** 指定しないとエンジンの入力が 640×640 固定になり、人物マスクも 640×640 で返ってきます。プログラムはマスクを 640×480 に単純にリサイズするので、人物領域が縦に潰れてずれます。PC で再現したところ、正しい位置との IoU は 0.63 でした。
対策として、本体ではマスクと画像のサイズが違う場合に `[WARNING]` を表示します。計測に使ったエンジンは、メタデータの `imgsz` が `[480, 640]` であることを確認しました。全計測で警告は出ていません。

Jetson で変換するときに必要だったこと:

- venv から JetPack の `tensorrt`(システムの Python 側)を import できるようにする(今回は venv の `site-packages` にシンボリックリンクを作成)
- `onnx` と `onnxslim` をインストールする

### ② → ③ トラッカーを ByteTrack に変更

Ultralytics の既定のトラッカー(BoT-SORT、`botsort.yaml`)は `gmc_method: sparseOptFlow` になっています。カメラ自体の動きを補正するため、毎フレーム CPU でオプティカルフローを計算する設定です。固定カメラではこの補正が不要なので、補正を行わない ByteTrack に変えました。

トラッキング等は 13.20 ms から 3.23 ms になりました。カメラが動く用途では、BoT-SORT の補正が有効な場合があります。

### ③ → ④ 描画の二重処理をやめる

`result.plot()` が描いた人物マスクは、直後にプログラム側で同じ領域を塗りつぶすため、最終的な画像には残っていませんでした。`result.plot(masks=False)` でマスクの描画を省きました。

- テスト画像で最終画像を比較し、307,200 画素中、違う画素が 0 であることを確認しました。
- 描画の時間は人物の映り方で変わるため、同じ姿勢のまま③と④を交互に計測しました。

| 順番 | 設定 | マスク処理+距離計算 | 描画 | 処理合計 | ログ |
|---|---|---|---|---|---|
| 1 | ③ | 2.27 | 10.49 | 26.37 | `ab1_bytetrack.txt` |
| 2 | ④ | 2.30 | 8.05 | 23.78 | `ab2_fastdraw.txt` |
| 3 | ③ | 2.30 | 10.52 | 26.58 | `ab3_bytetrack.txt` |
| 4 | ④ | 2.13 | 7.52 | 23.14 | `ab4_fastdraw.txt` |

描画は約 10.5 ms から約 7.8 ms になりました(約 −2.7 ms)。

## 参考:電源モードの影響

改善の前に、電源モードを 25W から MAXN_SUPER に変えたところ、処理合計(PyTorch)は 92.6 ms から 60.1 ms になりました。この 2 つの値は、計測コードを整理する前の版で計測したものです。区間の定義は同じで、MAXN_SUPER での値は①(61.06 ms)とほぼ一致しています。

## 既知の問題

- TensorRT エンジンの読み込みが、計測中 11 回のうち 2 回失敗しました(`cudaErrorStreamCaptureInvalidated`。Ultralytics の TensorRT 読み込み処理の中で発生)。どちらも、もう一度起動すると正常に動きました。原因は未調査です(`3_bytetrack_failed.txt`、`ab2_fastdraw_failed.txt`)。
- 距離ラベルの日本語は `cv2.putText` が対応していないため、正しく表示されません。
- 計測したのは 1 人の場合だけです。

ログ中のパスは、公開用に `<venv>`、`<repo>`、`~` に置き換えています。
