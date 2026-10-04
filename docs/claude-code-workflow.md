# Claude Code を使った進め方

このリポジトリのコードと計測は、AI エージェント Claude Code(ホスト PC 上で動作)を使って進めました。
人と AI の役割分担と、ホスト PC から Jetson を操作した方法を記録します。

![ホスト PC の Claude Code が SSH で Jetson を操作する流れ](images/ssh_workflow.png)

## 構成

```
ホスト PC(Ubuntu、RTX 5080)
  └─ VS Code + Claude Code
       ├─ PC 上でコードを編集・テスト
       └─ SSH で Jetson を操作(コードの配置、計測の実行、結果の回収)

Jetson Orin Nano + RealSense D435i
```

## SSH で Jetson を操作した方法

### 準備

```bash
# ホスト PC で実行する
# 鍵の作成
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_jetson -N ""
# 公開鍵を Jetson へ保存
ssh-copy-id -i ~/.ssh/id_ed25519_jetson.pub <user>@<jetson-ip>
```

- **Jetson のパスワードは AI に渡していません。** 会話の記録に残るのを避けるためです。また、AI のコマンド実行ではパスワードを対話的に入力できないため、鍵での接続にしました。
- この鍵は Jetson 専用にし、パスフレーズなしにしています(AI のコマンド実行から使うため)。GitHub 用の鍵とは分けています。
