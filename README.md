# Artificial Life Simulator — NEAT + Fission + Diverse Reseed (Spec)

- **ファイル**: `evo_sim_neat_diverse.py`
- **最終更新**: 2025-10-22
- **要約**: 2000×2000 のトーラス世界で、NEAT 風の可変構造脳を持つ個体が移動・捕食・交配・分裂する人工生命シミュレーター。絶滅時は最後の 10 個体をクラスタリングし、各クラスタから均等サンプリング＋微小変異で `N_INIT` 体を再播種して継続観察する。

---

## 1. 世界 (World)

- サイズ: `W=2000`, `H=2000`（トーラス境界）
- 時間刻み: `DT=0.15`
- 空間分割: グリッドセル `CELL=40` による近傍検索（`GRID_W`, `GRID_H`）
- 初期個体数: `N_INIT=50`
- 初期フード一括投入:
  - `INITIAL_FOOD_PIECES=2000` を面積スケールしてばら撒き
  - `INITIAL_FOOD_SCALE=True` なら `(W×H)/(2000×2000)` を乗じて自動調整
- ランタイムのフード出現: 毎 tick `FOOD_RATE=0.05` の確率で 1 個出現
- フード量: `FOOD_EN=50`（1 ピースのエネルギー）
- 死体: 殺害／自然死で `Body` を生成。エネルギー `BODY_INIT_EN=80`、毎 tick `DECAY_BODY=0.995` で減衰

---

## 2. 個体 (Agent)

- 状態: 位置 `(x, y)`, 速度 `(vx, vy)`, 体格 `S∈[0.8, 1.2]`, エネルギー `E`, 脳 (Genome)
- 行動出力（連続／離散混合）:
  - `thrust`（推進, `tanh`）
  - `turn`（旋回, `tanh×0.3`）
  - `attack`（攻撃トグル）
  - `mate`（交配トグル）
  - `eat_strength`（摂食強度, `[0..1]`）
- 速度上限: `vmax = SPEED_MAX_BASE * (1.2 - 0.2 * S)`（体格が大きいほど遅い）

### 2.1 センシング

- 視覚レイ: `N_RAYS=5` 本、各レイで近傍（他個体）の距離、相対サイズ（S 比）、相対速度投影を取得（上限距離 `R_SENSE=180`）
- 自己状態入力: `[E 正規化, S 正規化, 速度比, ダミー]`
- 入力次元: `IN_FEATURES = N_RAYS * 3 + 4`

---

## 3. 脳 (NEAT 風の可変構造)

- 表現: 有向非巡回グラフ (DAG)
- ノード: `NodeGene(id, type, bias)`（`type ∈ {INPUT, HIDDEN, OUTPUT}`）
- 結合: `ConnGene(innov, in_id, out_id, w, enabled)`
- イノベーション番号は `(in_id, out_id)` ペアで一意（`InnovationDB`）
- 初期化: 入力→出力の疎な結合をランダム生成
- 推論: トポロジ順序をキャッシュ（Kahn 法）。中間は `tanh`、出力は線形（行動側で `tanh` などに変換）
- 変異:
  - 構造: `add_connection`, `add_node`, `del_connection`（確率は `P_ADD_CONN=0.20`, `P_ADD_NODE=0.08`, `P_DEL_CONN=0.02`）
  - 重み／バイアス: ガウス摂動（`WEIGHT_SIGMA=0.08`, `WEIGHT_DRIFT_P=0.8`）
  - 微小変異（再播種／分裂用）: `micro_mutate(weight_sigma=0.03, p_add_conn=0.04, p_add_node=0.015, p_del_conn=0.008)`
- 交叉: イノベーション番号整列で親から混合

**脳コスト**: `BRAIN_COST_PER_CONN=0.0006` ×（有効結合数）を毎 tick のエネルギー消費に加算

---

## 4. 代謝・エネルギー

- 移動コスト: `MOVE_COST_K * speed^2`（`MOVE_COST_K=0.001`）
- 脳コスト: 上記の結合数比例
- 基礎代謝: `BASE_COST=0.35`（停止していても減る）
- 飢餓コスト: `E < STARVATION_E (=120)` のとき `STARVATION_COST=0.55`
- アイドル・ペナルティ:
  - `spd < IDLE_SPEED_FRAC * vmax`（`IDLE_SPEED_FRAC=0.10`）かつ
  - `|thrust| < 0.2` 且つ `|turn| < 0.05` のとき `IDLE_COST=0.45` 追加

目的: 止まっていると強く不利、空腹時ほど寿命短縮。

---

## 5. 相互作用

- 摂食:
  - フード: 距離 `< R_HIT=12` で `take = min(f.e, FOOD_EN) * (0.3 + 0.7 * eat_strength)`
  - 死体: 同様に `take = min(body.e, 10.0 * (0.3 + 0.7 * eat_strength))`
  - 摂食量は分裂判定のカウンタ `eaten_units` に加算
- 交配（有性）:
  - 互いに `mate` が真、距離 `< R_HIT`、両者 `E > E_BIRTH_THRESHOLD=220`、相性 `is_compatible(a, b)` を満たす
  - 子の Genome は交叉＋構造／重み変異、子 `E=CHILD_EN=120`、親は `PARENT_COST=80` 消費
- 攻撃（捕食）:
  - `attack` 真かつ `8.0 * a.S > 5.0 * b.S` で殺害成功、攻撃者 `E += 60`、死体を生成
- 分裂（無性）:
  - `E > FISSION_ENERGY_TH=280` かつ `eaten_units >= FISSION_FOOD_UNITS_TH=5`
  - 親のクローンに微小変異を適用して子生成（子 `E=100`、親 `-70`、カウンタリセット）

---

## 6. 絶滅セーフティ & 多様性再播種

- 保存: 実行中に約 10 秒間隔で上位 10 個体（`E` の高い順）を `last10_genomes.json` に保存
- 絶滅検知: 全個体消滅時
- 再播種:
  1. `last10` を読み込み、各 Genome を特徴ベクトル化（64 次元ハッシュ + 6 統計 = 70 次元, L2 正規化）
  2. `K-means`（`2〜min(5, n)` クラスタ）
     - `K-means++` 初期化を数値安定化
     - 同一点群／少数データにも対応
     - 空クラスタはランダム再初期化
  3. 各クラスタから均等ラウンドロビンでサンプリング → サンプリング個体のゲノムは微小変異して `N_INIT` 体を生成
- ロード操作: `L` キーで手動リシード（クラスタ均等）

---

## 7. 可視化 & 操作

- 表示: `pygame`
- 個体は体格 `S` とエネルギーで色／サイズが変化、向きに短いベクトルを描画
- フードは緑、死体は茶色
- HUD: 時刻 `t`、個体数 `n`、平均エネルギー `meanE`、出生／死亡カウンタ、FPS、操作ガイド
- キー操作:
  - `SPACE`: 一時停止／再開
  - `F`: 高速モード（`1 tick → 6 substeps`）
  - `S`: スクリーンショット保存（`screens/`）
  - `R`: ランダム初期化
  - `L`: `last10` から多様性リシード
  - `Q` / `ESC`: 終了

---

## 8. 主要パラメータ（チューニング指針）

| 目的 | パラメータ | 推奨調整 |
| --- | --- | --- |
| 初期の餌不足解消 | `INITIAL_FOOD_PIECES`, `INITIAL_FOOD_SCALE` | `2000→3000` など増量 |
| 継続的な餌密度 | `FOOD_RATE` | `0.05→0.08` で湧き増 |
| 1 個あたり回復量 | `FOOD_EN` | `50→80` で強化 |
| 停止を不利に | `BASE_COST`, `IDLE_COST` | `+0.1` 刻みで上げる |
| 空腹時の寿命短縮 | `STARVATION_E`, `STARVATION_COST` | `E` 域を広げる／消費増やす |
| 脳の肥大抑制 | `BRAIN_COST_PER_CONN` | `0.0006→0.0010` |
| 高速ダラダラ抑制 | `MOVE_COST_K` | 速度² 比例、`0.001→0.006` |

---

## 9. 保存フォーマット

`last10_genomes.json`:

```json
[
  {
    "genome": {
      "nodes": [{"id": int, "type": 0 | 1 | 2, "bias": float}, ...],
      "conns": [{"innov": int, "in_id": int, "out_id": int, "w": float, "enabled": bool}, ...],
      "in_ids": [int, ...],
      "out_ids": [int, ...]
    },
    "S": float
  },
  ...
]
```

---

## 10. 再現性・性能

- 乱数: `numpy.default_rng(1)` と `random.Random(1)` を使用（完全固定ではないが近い）
- 近傍探索: グリッドで疑似 `O(1)` 近傍
- 推論: 変異時のみトポロジ再構成、通常はキャッシュ

---

## 11. 拡張案

- 入力拡張: フード／死体専用レイ、群集密度、履歴ベクトル
- 表示強化: クラスタごとに色分け、行動ヒートマップ
- 制約: `STALL_KILL_TICKS`（長時間アイドル即死）
- 遺伝子拡張: 出力増（例えば探索／回避モード切替）

---

## 12. トラブルシュート

- `K-means` 初期化エラー（確率が 1 にならない）
  - 実装側でゼロ和／NaN を一様分布にフォールバック済み
- 初期で餓死が多い
  - `INITIAL_FOOD_PIECES` 増、`FOOD_RATE` 増、`FOOD_EN` 増
- 止まりがち
  - `BASE_COST` / `IDLE_COST` / `STARVATION_COST` の順に上げる
- 頻繁に絶滅
  - フード増、代謝弱め、分裂閾値引き下げ

---

## 13. 実行

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate
pip install pygame
python evo_sim_neat_diverse.py
```
