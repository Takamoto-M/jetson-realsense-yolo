# Claude Code を使った進め方

このリポジトリのコードと計測は、AI エージェント Claude Code(ホスト PC 上で動作)を使って進めました。
人と AI の役割分担と、ホスト PC から Jetson を操作した方法を記録します。

## 構成

```
ホスト PC(Ubuntu、RTX 5080)
  └─ VS Code + Claude Code
       ├─ PC 上でコードを編集・テスト
       └─ SSH で Jetson を操作(コードの配置、計測の実行、結果の回収)

Jetson Orin Nano + RealSense D435i
```

## 進めた順番

| 段階 | 内容 | 実行したのは |
|---|---|---|
| 1 | 計測の仕組みを作る。PC 上で、実カメラ(人物なし)と、`pyrealsense2` を差し替えた模擬カメラ(テスト画像で人物あり)を使って動作確認 | Claude Code |
| 2 | Jetson で計測する。電源モードの変更(25W → MAXN_SUPER) | 筆者(コードは USB で受け渡し) |
| 3 | TensorRT への変換。`onnx` のインストール、venv から `tensorrt` を使えるようにする対処 | 筆者(Claude Code が手順を案内) |
| 4 | マスクのずれに気づき、PC で再現して原因を特定。`--model` 引数と警告表示を追加 | Claude Code |
| 5 | 計測コードを簡易版にし、改善用のオプション(`--tracker`、`--fast-draw`)を追加 | Claude Code |
| 6 | SSH の鍵を作り、Jetson に登録 | 筆者 |
| 7 | SSH で Jetson の状態を確認(エンジンのメタデータの確認を含む)、コードを配置、計測①〜④と交互計測を実行、結果を回収 | Claude Code |
| 8 | `sudo jetson_clocks`、カメラの前に立つ、画面で結果の正しさを確認 | 筆者 |

方針の決定(何を測るか、どの改善を試すか、記事をどう見せるか)は、筆者が行いました。

## SSH で Jetson を操作した方法

### 準備(筆者が実行)

```bash
# ホスト PC で実行する(Jetson につないだ状態のターミナルで実行しないこと)
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_jetson -N ""
ssh-copy-id -i ~/.ssh/id_ed25519_jetson.pub <user>@<jetson-ip>
```

- **Jetson のパスワードは AI に渡していません。** 会話の記録に残るのを避けるためです。また、AI のコマンド実行ではパスワードを対話的に入力できないため、鍵での接続にしました。
- この鍵は Jetson 専用にし、パスフレーズなしにしています(AI のコマンド実行から使うため)。GitHub 用の鍵とは分けています。
- 鍵はホスト PC で作る必要があります。実行前に、プロンプトのホスト名がホスト PC になっていることを確認してください。

### Claude Code が実行したコマンドの形

```bash
J="ssh -i ~/.ssh/id_ed25519_jetson -o BatchMode=yes <user>@<jetson-ip>"
$J 'cd ~/source/jetson-realsense-yolo && source ~/venv/venv_py310/bin/activate && \
    python3 realsense_yolo_depth.py --profile --no-display --max-frames 600 ... > results/xx.txt'
scp -i ~/.ssh/id_ed25519_jetson '<user>@<jetson-ip>:source/jetson-realsense-yolo/results/*.txt' ./
```

- `BatchMode=yes` を付け、パスワードを聞かれる状況では待たずにエラーにしています。
- 計測には必ず `--max-frames` を付け、自動で終了するようにしました。
- 画面表示ありの実行は、`DISPLAY=:0` を指定して Jetson のモニターに表示しました。

### 守ったルール

- Jetson の既存の作業フォルダには触らず、新しいフォルダにコードを置きました。モデルとエンジンは、既存のものへのシンボリックリンクにしました。
- `sudo` が必要な操作は、筆者が実行しました。
- Jetson でコマンドを実行する前に、何をするかを説明しました。
- GitHub への push など外部に影響する操作は、筆者の確認後に行うことにしました。

## AI に任せられたこと・任せられなかったこと

| 任せられたこと | 任せられなかったこと |
|---|---|
| コードの作成とテスト(模擬カメラを含む) | カメラの前に立つ(人物がいないと計測できない) |
| 計測の実行と、結果の比較・集計 | 画面で、マスクの位置や追跡の様子を目で確認する |
| エラーの原因調査(マスクのずれを PC で再現など) | `sudo` が必要な操作、パスワードの入力 |
| ログの公開前の整理(パスの除去) | 何を目的にし、どの案を選ぶかの判断 |

計測中のトラブル(TensorRT エンジンの読み込みエラー)は、Claude Code がログを確認し、再実行して対処しました。
