# NEAT シミュレーション本体仕様

本ドキュメントは GUI アプリケーションが現在採用している NEAT ベース人工生命シミュレーション (`app/sim/neat_simulation.py`) の仕組みと主なパラメータをまとめたものです。  
実装は `old/evo_sim_neat_diverse.py` をヘッドレス用途に再構成し、GUI から設定可能な要素を中心に整理しています。

---

## 1. 全体構成
- **World**  
  - トーラス（幅 `W` × 高さ `H`）上で個体・フード・死体を管理。  
  - グリッド分割 (`CELL`) による近傍探索で相互作用を高速化。  
  - `tick(substeps)` がシミュレーションのメインループ。
- **Agent**  
  - 位置・速度・体格 (`S`)・エネルギー (`E`) と可変構造脳 (`Genome`) を持つ。  
  - `step()` で入力センサから出力行動を決定し、移動・代謝などを処理。  
  - 食餌、交配、攻撃、無性分裂の状態を保持（`eat_strength`/`mate`/`attack` など）。
- **Genome / NEAT 風可変構造脳**  
  - ノード (`NodeGene`) と結合 (`ConnGene`) による有向非巡回グラフ。  
  - Innovation 番号による整列交叉、結合/ノード追加・削除、重み摂動。  
  - 入力ノード数: `IN_FEATURES = N_RAYS*3 + 4`（レイセンサー + 自己状態）。  
  - 出力ノード数: `OUT_FEATURES = 5`（推進・旋回・攻撃・交配・摂食強度）。
- **保存 & 再播種**  
  - 定期的にエネルギー上位 `SAVE_TOP_K` 個体を `last10_genomes.json` へ保存。  
  - 全滅時は保存された個体を特徴空間でクラスタリングし、均等サンプリング + 微小変異で `N_INIT` 体を再生成。

---

## 2. 環境モデル
| 項目 | 説明 | 主パラメータ |
| ---- | ---- | ------------- |
| フード出現 | `FOOD_RATE` を基準に `spawn_food` で確率的に追加。フード 1 個あたり `FOOD_EN` エネルギーを付与。 | `FOOD_RATE`, `FOOD_EN` |
| マルチタイプフード | `FOOD_TYPE_ENERGIES` と `FOOD_TYPE_WEIGHTS` に基づき 5 種類のフードを生成。高栄養ほど出現確率が低くなる。エージェントは種別のみ識別可能（エネルギー量は未知）。 | `food_type_energies`, `food_type_weights` |
| 初期フード | 起動直後は `INITIAL_FOOD_PIECES` を一括投入。`INITIAL_FOOD_SCALE=True` なら面積比で自動スケール。 | `INITIAL_FOOD_PIECES`, `INITIAL_FOOD_SCALE` |
| 密度フィールド | `FOOD_DENSITY_VARIATION` のσを用いた指数乱数で 2D 密度マップを生成。平均 1 に正規化し、CDF からスポーン位置を決定。 | `FOOD_DENSITY_VARIATION` |
| 動的季節変動 | `SEASON_PERIOD`、`SEASON_AMPLITUDE` により `FOOD_RATE` を周期的に変調。`season_multiplier = 1 + A sin(2πt/T)`。 | `SEASON_PERIOD`, `SEASON_AMPLITUDE` |
| 危険フィールド | カバレッジ比率 `HAZARD_COVERAGE` と強度 `HAZARD_STRENGTH` に基づきランダムマップを生成。セルに滞在すると `hazard * ダメージ係数` を毎 tick 消費。 | `HAZARD_STRENGTH`, `HAZARD_COVERAGE` |
| 死体減衰 | `DECAY_BODY` に従いエネルギーが指数減衰。一定以下で削除。 | `DECAY_BODY`, `BODY_INIT_EN` |

季節や危険フィールドは GUI の「環境」タブから設定可能です。

---

## 3. 個体と行動
### 3.1 センサー
- **レイベース視覚 (`N_RAYS`)**  
  - 前方 ±60° に等角レイを飛ばし、最も近い対象との距離 / サイズ比 / 相対速度投影を検知。  
  - `R_SENSE` が最大感知距離。
- **自己状態**  
  - エネルギー (`E/400`)、体格 (`S`)、速度比、ダミー入力の 4 要素。
- **フードヒント**  
  - 近傍で最も近いフードの「種類」と「距離」を追加入力として提供。種類は 0～1 に正規化されたラベルで、栄養価そのものは観測できない。

### 3.2 出力
- `thrust`（推進）  
- `turn`（旋回）  
- `attack`（攻撃フラグ）  
- `mate`（交配フラグ）  
- `eat_strength`（摂食強度）  

（将来的にロードマップ Stage 2 で追加行動を拡張予定）

### 3.3 代謝モデル
| コスト | 計算式 | パラメータ |
| ------ | ------ | --------- |
| 移動 | `MOVE_COST_K * speed^2` | `MOVE_COST_K` |
| 脳コスト | 有効結合数 × `BRAIN_COST_PER_CONN` | `BRAIN_COST_PER_CONN` |
| 基礎代謝 | 常時 `BASE_COST` 消費 | `BASE_COST` |
| 飢餓 | `E < STARVATION_E` のとき `STARVATION_COST` 追加 | `STARVATION_E`, `STARVATION_COST` |
| アイドル | 速度と出力が閾値以下のとき `IDLE_COST` 追加 | `IDLE_SPEED_FRAC`, `IDLE_THRUST_TH`, `IDLE_TURN_TH`, `IDLE_COST` |
| 危険ダメージ | 危険セル滞在時 `hazard * damage_scale` を減算 | `HAZARD_STRENGTH`, `HAZARD_COVERAGE` |

### 3.4 分裂（無性）
- 条件: `E > FISSION_ENERGY_TH / fission_bias` かつ `食餌累積 ≥ FISSION_FOOD_UNITS_TH / fission_bias`  
- 親は `FISSION_PARENT_COST / fission_bias` を消費し、子は `FISSION_CHILD_EN × min(2, fission_bias)` のエネルギーで誕生。  
- `fission_bias` は GUI の「代謝」タブから変更でき、分裂のしやすさを一括調整可能。

### 3.5 有性交配
- `mate` が双方 true、距離 < `R_HIT`、`E > E_BIRTH_THRESHOLD`、`is_compatible` 判定を満たす場合に成立。  
- 子は交叉＋構造変異後にエネルギー `CHILD_EN` を得る。親は `PARENT_COST` を消費。

### 3.6 攻撃
- 攻撃者 `attack` true で `8 * S_attacker > 5 * S_target` のとき成功。  
- 攻撃成功でエネルギー+60、ターゲットは死亡し死体 (`BODY_INIT_EN`) を残す。

---

## 4. 脳（NEAT 風可変構造）
- `Genome` は入力・出力 ID を保持し、途中でノード追加時は `NEXT_NODE_ID` を採番。  
- 推論時はトポロジカル順序をキャッシュ（Kahn 法）し、`forward()` で順に計算。  
- 変異パラメータ: `P_ADD_CONN`, `P_ADD_NODE`, `P_DEL_CONN`, `WEIGHT_SIGMA`, `WEIGHT_DRIFT_P`。
- 微小変異 (`micro_mutate`) は再播種・分裂時に使用し、より控えめな確率で構造変更。
- `InnovationDB` が `(in_id, out_id)` ペアから一意のイノベーション番号を割り振り、交叉時に結合を整列させる。

---

## 5. 多様性再播種
1. `last10_genomes.json` から最大 10 個の個体を読み込み。  
2. 各ゲノムを 64 次元ハッシュ + 6 次元統計で特徴ベクトル化。  
3. K-means（`KMEANS_MAX_K` 以下）でクラスタリング。  
4. 各クラスタからラウンドロビンでサンプリングし、微小変異で `N_INIT` 体を生成。  
5. `FISSION_FOOD_UNITS_TH` や `FISSION_ENERGY_TH` は `fission_bias` によって自動調整。

---

## 6. 主な定数一覧
| 区分 | パラメータ | 既定値 | 説明 |
| ---- | ---------- | ------ | ---- |
| 環境 | `W`, `H` | 2000 × 2000 | ワールドサイズ |
| 初期設定 | `N_INIT` | 50 | 初期個体数 |
| 時間刻み | `DT` | 0.15 | 1 tick あたりの時間 |
| センサー | `N_RAYS` | 5 | レイの本数 |
| 交互作用 | `R_SENSE`, `R_HIT` | 180, 12 | 感知距離/相互作用距離 |
| 変異 | `P_ADD_CONN`, `P_ADD_NODE`, `P_DEL_CONN` | 0.20 / 0.08 / 0.02 | 構造変異確率 |
| 季節 | `SEASON_PERIOD`, `SEASON_AMPLITUDE` | 1200, 0.0 | フード出現率の周期変動 |
| 危険 | `HAZARD_STRENGTH`, `HAZARD_COVERAGE` | 0.0, 0.0 | 危険フィールド生成パラメータ |
| 分裂バイアス | `FISSION_RATE_FACTOR` | 1.0 | GUI の分裂バイアス値と連動 |

---

## 7. 拡張余地
- 行動出力の追加（ダッシュ、防御、休息など）  
- 形質（HP/弾性/代謝タイプ）の遺伝子化  
- 入力センサ拡張（危険濃度、履歴ベクトル）  
- 多様性維持（ニッチシェアリング、老化、新奇性）  
- プリセットシナリオ（砂漠・湿地・季節など）

詳細な拡張計画は [`docs/roadmap.md`](./roadmap.md) を参照してください。
