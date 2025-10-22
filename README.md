# Artificial Life Simulator GUI

人工生命シミュレーションを操作・観察するための PySide6 製 GUI アプリケーションです。  
NEAT 風の可変構造脳を持つ個体シミュレーション（`app/sim/neat_simulation.py`）をヘッドレスで動かしながら、パラメータ変更・可視化・データ保存／読み込みを GUI から行えます。  
`old/` ディレクトリには旧実装が残っていますが、今後は `app/sim/neat_simulation.py` を利用してください。

---

## 主な特徴
- **リアルタイム表示**: 個体・餌・死体の配置をトップダウンで描画するワールドビュー、統計情報（個体数・平均エネルギー・出生／死亡数・資源数）をリアルタイムで表示。
- **シミュレーション制御**: GUI から開始／停止・スナップショット取得が可能。NEAT 実装とダミー実装（スタブ）をコマンドラインで切り替え可能。
- **設定編集**: World / Resources / Metabolism / Brain の各セクションを GUI 上で編集。セクションごとに JSON として保存・読み込みでき、前回使用したファイルパスを記憶します。
- **エージェント保存／読み込み**: `last10_genomes.json` 相当の形式で任意の場所に保存・読み込みが可能。読み込んだファイルのパスも記憶し、次回起動時にサジェストします。
- **状態永続化**: `.alsim_state.json` に最後に使用したファイルパスを保存し、次回起動時に自動復元。

---

## ディレクトリ構成
```
.
├── app/                     # GUI アプリケーション
│   ├── main.py              # エントリーポイント
│   ├── core/                # バックエンド・設定関連
│   ├── sim/                 # NEAT シミュレーション本体
│   ├── ui/                  # Qt ウィジェット
│   └── state_manager.py     # ファイルパス永続化
├── old/                     # 旧シミュレーション実装（移行期間用）
│   └── evo_sim_neat_diverse.py
├── last10_genomes.json      # 直近保存したエージェント情報（例）
├── *_config.json            # セクションごとの設定保存例
└── README.md
```

---

## 前提・依存パッケージ
```bash
pip install PySide6 numpy
```
※ 旧ビューア（`old/evo_sim_neat_diverse.py`）を起動する場合のみ `pygame` が必要です。

---

## 起動方法

### 1. スタブ（ダミー）バックエンドで試す場合
```bash
python -m app.main --backend stub
```
スタブは疑似的に統計値・ワールドビューを生成するため、依存ライブラリが少なく環境確認に便利です。

### 2. NEAT シミュレーションを GUI で動かす場合
```bash
python -m app.main --backend neat
```
デフォルトでは `app.sim.neat_simulation` が読み込まれます。  
別の実装を試したい場合は下記のように `--sim-module` や `--sim-file` を指定してください。

```bash
# カスタムモジュール名を使う場合
python -m app.main --sim-module mypkg.custom_sim

# 単独の .py ファイルを指定する場合
python -m app.main --sim-file path/to/custom_sim.py
```

#### 主なコマンドラインオプション
| オプション | 説明 | 既定値 |
| ---------- | ---- | ------ |
| `--backend {neat,stub}` | 使用するバックエンド種別 | `neat` |
| `--sim-module NAME` | インポートするモジュール名 | `evo_sim_neat_diverse` |
| `--sim-file PATH` | モジュール ファイルへの絶対／相対パス | なし |
| `--tick-substeps N` | 1 GUI チックあたりのシミュレーション進行サブステップ | 1 |
| `--sleep-interval SEC` | バックエンドの `step()` 呼び出し間隔秒 | 0.01 |

---

## GUI の使い方

### メイン画面
- ステータスバーに実行状態（Running/Stopped）を表示。
- 左ペインで設定編集、右ペインで世界表示・統計・操作ボタン／ログを配置。
- `Start` / `Stop` ボタンでシミュレーション制御。
- `Save Agents` / `Load Agents` でエージェント情報を保存／復元。

### ワールドビュー
- 黒背景に緑（餌）、茶色（死体）、色付き円（個体）を描画。
- 個体の色はエネルギー量で変化し、サイズは体格でスケーリング。

### 設定エディタ
- `World` `Resources` `Metabolism` `Brain & Mutation` の 4 セクション。
- 各セクションに `Save…` / `Load…` ボタンを配置し、JSON ファイルとの入出力が可能。
- 例: `world_config.json`, `resources_config.json` など。

---

## 保存フォーマット

### 1. エージェント (`last10_genomes.json`)
`evo_sim_neat_diverse.py` の実装と同じ形式で保存されます（出生エネルギーの高い上位個体を 10 件保存）。

```json
[
  {
    "genome": {
      "nodes": [{"id": 0, "type": 0, "bias": 0.0}, ...],
      "conns": [{"innov": 1, "in_id": 0, "out_id": 100, "w": 0.42, "enabled": true}, ...],
      "in_ids": [0,1,2,...],
      "out_ids": [100,101,...]
    },
    "S": 1.03
  },
  ...
]
```

NEAT バックエンドで読み込む際は `_bootstrap_from_last10_diverse` が自動で呼び出され、クラスタリング後に個体を再配置します。

### 2. セクション設定 (`*_config.json`)
各セクションごとの値を含むシンプルな JSON オブジェクトです。例（World 設定）:
```json
{
  "width": 2000.0,
  "height": 2000.0,
  "time_step": 0.15,
  "initial_population": 50
}
```

---

## 永続化されるファイル

| ファイル | 役割 |
| -------- | ---- |
| `.alsim_state.json` | 最後に使用したエージェント／設定ファイルのパスを保存。 |
| `last10_genomes.json` | `Save Agents` ボタンの既定候補。 |
| `*_config.json` | セクション別の設定バックアップ例。 |

`.alsim_state.json` はアプリ終了時もしくは保存操作時に随時更新され、次回起動時に自動的に読み込まれます。

---

## 開発メモ
- GUI は PySide6 を前提にしているため、macOS / Windows / Linux いずれでも Qt ランタイムが利用可能。
- バックエンドは `SimulationBackend` プロトコルに従って実装し、`step()` / `snapshot()` / `save_agents()` / `load_agents()` を備えれば GUI と連携できます。
- `StubSimulationBackend` は UI 検証用のサンプル実装であり、実際のシミュレーションロジックは `NeatSimulationBackend` を介して `evo_sim_neat_diverse.py` と連携します。

---

## トラブルシュート
- **ModuleNotFoundError: `evo_sim_neat_diverse`**  
  `--sim-file old/evo_sim_neat_diverse.py` を指定するか、モジュールを PYTHONPATH に追加してください。
- **pygame の初期化エラー**  
  NEAT バックエンドでは pygame をインポートします。GUI 実行中はヘッドレスで動作するように設定されていますが、pygame がインストールされていることを確認してください。
- **設定ファイルに無効な値を入れた**  
  `Load…` 操作で不正な JSON を読み込んだ場合、エラーダイアログが表示されます。既定値に戻したい場合はアプリを再起動するか、適切な JSON を読み込み直してください。

---

## ライセンス
特に明記していない限り、本リポジトリ内のコードは MIT ライセンスを前提としています（`app/*.py` の各ファイル先頭参照）。シミュレーションロジックの一部は元実装に従います。
