# SPDX-License-Identifier: MIT
"""
Headless NEAT-based artificial life simulation used by the GUI backend.

This module is derived from the original `old/evo_sim_neat_diverse.py`, with
all pygame viewer code removed so it can run purely as a simulation service.
"""

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ===================== 基本環境パラメータ =====================
W, H            = 2000.0, 2000.0
N_INIT          = 50
DT              = 0.15
SPEED_MAX_BASE  = 8.0
R_SENSE         = 180.0
VISION_DEG      = 180.0
R_HIT           = 12.0

FOOD_RATE       = 0.05#0.015
FOOD_EN         = 50.0
# --- 初期フードシード量（起動直後の欠食対策） ---
INITIAL_FOOD_PIECES    = 2000#1200   # 起動時にばら撒くフード個数（W=H=2000想定）。密度を上げたい場合は増やす。
INITIAL_FOOD_SCALE     = True   # True の場合、ワールド面積に応じて自動スケール
DECAY_BODY      = 0.995
BODY_INIT_EN    = 80.0

MOVE_COST_K             = 0.001#0.002
BRAIN_COST_PER_CONN     = 0.0006#0.0006  # ← 結合数比例

# --- 新規追加（基礎代謝 & 飢餓 & アイドル・ペナルティ） ---
BASE_COST               = 0.35       # 毎tickで必ず払う（止まっていても減る）
STARVATION_E            = 120.0      # ここを下回ると飢餓モード
STARVATION_COST         = 0.55       # 飢餓モード時の追加コスト（毎tick）
IDLE_SPEED_FRAC         = 0.10       # vmaxの10%未満を「低速」とみなす
IDLE_THRUST_TH          = 0.2        # 出力が弱い＝意図的に動いていない/詰まってる
IDLE_TURN_TH            = 0.05
IDLE_COST               = 0.45       # アイドル・ペナルティ（毎tick）

E_BIRTH_THRESHOLD       = 220.0
PARENT_COST             = 80.0
CHILD_EN                = 120.0

# 分裂（無性）条件
FISSION_ENERGY_TH       = 280.0
FISSION_FOOD_UNITS_TH   = 5.0     # “食べた量”の累積しきい値
FISSION_CHILD_EN        = 100.0
FISSION_PARENT_COST     = 70.0

# 構造変異（NEAT風）
P_ADD_CONN      = 0.20
P_ADD_NODE      = 0.08
P_DEL_CONN      = 0.02
WEIGHT_SIGMA    = 0.08
WEIGHT_DRIFT_P  = 0.8   # 既存重みを摂動 vs 再初期化

# 近親/適合度距離（繁殖相性）
ASSORT_ALPHA    = 0.5
COMP_TH         = 1.5

# 空間分割
CELL            = 40.0
GRID_W          = int(math.ceil(W / CELL))
GRID_H          = int(math.ceil(H / CELL))

# 保存/復元
LAST10_PATH     = str((Path(__file__).resolve().parent.parent.parent / "last10_genomes.json"))
SAVE_TOP_K      = 10

# 多様性ブートストラップ
KMEANS_DIM      = 64   # ハッシュ特徴次元
KMEANS_MAX_K    = 5    # last10<=10 のためクラスタ数上限
KMEANS_ITERS    = 50
KMEANS_TRIES    = 6    # 初期化マルチトライ（ベストSSE）

rng  = np.random.default_rng(1)
rand = random.Random(1)

# ===================== ユーティリティ =====================
def wrap(v, L):
    if v < 0: return v + L
    if v >= L: return v - L
    return v

def torus_delta(dx, L):
    if dx >  L/2: dx -= L
    if dx < -L/2: dx += L
    return dx

def hsl_to_rgb(h, s, l):
    import colorsys
    r,g,b = colorsys.hls_to_rgb(h, l, s)
    return int(r*255), int(g*255), int(b*255)

# ===================== NEAT 風遺伝子表現 =====================
INPUT, HIDDEN, OUTPUT = 0, 1, 2

@dataclass
class NodeGene:
    id: int
    type: int
    bias: float = 0.0

@dataclass
class ConnGene:
    innov: int       # イノベーション番号
    in_id: int
    out_id: int
    w: float
    enabled: bool = True

class InnovationDB:
    """(in_id, out_id) -> unique innovation id"""
    def __init__(self):
        self.next_innov = 1
        self.map: Dict[Tuple[int,int], int] = {}

    def get_innov(self, in_id: int, out_id: int) -> int:
        k = (in_id, out_id)
        if k not in self.map:
            self.map[k] = self.next_innov
            self.next_innov += 1
        return self.map[k]

INNOV_DB = InnovationDB()
NEXT_NODE_ID = 1000  # 入出力以外はここから採番

class Genome:
    def __init__(self, in_ids: List[int], out_ids: List[int]):
        self.nodes: Dict[int, NodeGene] = {}
        self.conns: Dict[int, ConnGene] = {}
        for nid in in_ids:
            self.nodes[nid] = NodeGene(nid, INPUT, 0.0)
        for nid in out_ids:
            self.nodes[nid] = NodeGene(nid, OUTPUT, 0.0)
        self.topo_cache_valid = False
        self.topo_order: List[int] = []
        self.in_ids = list(in_ids)
        self.out_ids = list(out_ids)

    def clone(self) -> 'Genome':
        g = Genome(self.in_ids, self.out_ids)
        g.nodes = {nid: NodeGene(n.id, n.type, n.bias) for nid,n in self.nodes.items()}
        g.conns = {innov: ConnGene(c.innov, c.in_id, c.out_id, c.w, c.enabled) for innov,c in self.conns.items()}
        g.topo_cache_valid = False
        return g

    # ---------- 変異 ----------
    def mutate_weights(self):
        for c in self.conns.values():
            if rand.random() < WEIGHT_DRIFT_P:
                c.w += rng.normal(0, WEIGHT_SIGMA)
            else:
                c.w = rng.normal(0, 1.0)
        for n in self.nodes.values():
            n.bias += rng.normal(0, WEIGHT_SIGMA*0.5)
        self.topo_cache_valid = False

    def add_connection(self, tries=20):
        self._ensure_topo()
        if len(self.topo_order) < 2: return
        for _ in range(tries):
            a = rand.choice(self.topo_order)
            b = rand.choice(self.topo_order)
            if a == b: continue
            src, dst = (a,b) if self._topo_index(a) < self._topo_index(b) else (b,a)
            if self.nodes[src].type == OUTPUT:  continue
            if self.nodes[dst].type == INPUT:   continue
            if self._has_edge(src, dst):        continue
            innov = INNOV_DB.get_innov(src, dst)
            self.conns[innov] = ConnGene(innov, src, dst, rng.normal(0,1.0), True)
            self.topo_cache_valid = False
            return

    def add_node(self):
        enabled = [c for c in self.conns.values() if c.enabled]
        if not enabled: return
        edge = rand.choice(enabled)
        edge.enabled = False
        global NEXT_NODE_ID
        new_id = NEXT_NODE_ID; NEXT_NODE_ID += 1
        self.nodes[new_id] = NodeGene(new_id, HIDDEN, bias=0.0)
        innov1 = INNOV_DB.get_innov(edge.in_id, new_id)
        innov2 = INNOV_DB.get_innov(new_id, edge.out_id)
        self.conns[innov1] = ConnGene(innov1, edge.in_id, new_id, 1.0, True)
        self.conns[innov2] = ConnGene(innov2, new_id, edge.out_id, edge.w, True)
        self.topo_cache_valid = False

    def del_connection(self):
        if not self.conns: return
        innov = rand.choice(list(self.conns.keys()))
        del self.conns[innov]
        self.topo_cache_valid = False

    def structural_mutation(self):
        if rand.random() < P_ADD_CONN: self.add_connection()
        if rand.random() < P_ADD_NODE: self.add_node()
        if rand.random() < P_DEL_CONN: self.del_connection()
        self.mutate_weights()

    # ---------- 軽量（微小）変異 ----------
    def micro_mutate(self,
                     weight_sigma: float = 0.03,
                     p_add_conn: float = 0.05,
                     p_add_node: float = 0.02,
                     p_del_conn: float = 0.01) -> 'Genome':
        child = self.clone()
        if rand.random() < p_add_conn: child.add_connection()
        if rand.random() < p_add_node: child.add_node()
        if rand.random() < p_del_conn: child.del_connection()
        for c in child.conns.values():
            c.w += rng.normal(0, weight_sigma)
        for n in child.nodes.values():
            n.bias += rng.normal(0, weight_sigma * 0.5)
        child.topo_cache_valid = False
        return child

    # ---------- 交叉 ----------
    @staticmethod
    def crossover(ga:'Genome', gb:'Genome') -> 'Genome':
        child = ga.clone()
        child.conns.clear()
        set_a = set(ga.conns.keys())
        set_b = set(gb.conns.keys())
        all_innov = sorted(set_a | set_b)
        for innov in all_innov:
            gene = ga.conns.get(innov) if rand.random()<0.5 else gb.conns.get(innov)
            if gene is None:
                gene = ga.conns.get(innov) or gb.conns.get(innov)
            child.conns[innov] = ConnGene(
                innov=gene.innov, in_id=gene.in_id, out_id=gene.out_id,
                w=gene.w, enabled=gene.enabled
            )
            if gene.in_id not in child.nodes:
                src = (ga.nodes.get(gene.in_id) or gb.nodes.get(gene.in_id))
                child.nodes[gene.in_id] = NodeGene(src.id, src.type, src.bias)
            if gene.out_id not in child.nodes:
                dst = (ga.nodes.get(gene.out_id) or gb.nodes.get(gene.out_id))
                child.nodes[gene.out_id] = NodeGene(dst.id, dst.type, dst.bias)
        child.topo_cache_valid = False
        return child

    # ---------- 推論 ----------
    def forward(self, x_dict: Dict[int,float]) -> Dict[int,float]:
        self._ensure_topo()
        val: Dict[int,float] = {}
        for nid in self.topo_order:
            node = self.nodes[nid]
            if node.type == INPUT:
                val[nid] = x_dict.get(nid, 0.0)
                continue
            s = node.bias
            for c in self._incoming[nid]:
                if not c.enabled: continue
                s += val.get(c.in_id, 0.0) * c.w
            if node.type == OUTPUT:
                val[nid] = s
            else:
                val[nid] = math.tanh(s)
        return {nid: val[nid] for nid in self.out_ids}

    # ---------- 内部構造  ----------
    def _has_edge(self, u, v) -> bool:
        for c in self.conns.values():
            if c.in_id==u and c.out_id==v and c.enabled:
                return True
        return False

    def _topo_index(self, nid) -> int:
        if not self.topo_cache_valid: self._ensure_topo()
        return self._topo_pos.get(nid, 0)

    def _ensure_topo(self):
        if self.topo_cache_valid: return
        incoming: Dict[int, List[ConnGene]] = {nid: [] for nid in self.nodes.keys()}
        outgoing: Dict[int, List[ConnGene]] = {nid: [] for nid in self.nodes.keys()}
        indeg: Dict[int, int] = {nid: 0 for nid in self.nodes.keys()}
        for c in self.conns.values():
            if c.enabled and c.in_id in self.nodes and c.out_id in self.nodes:
                outgoing[c.in_id].append(c)
                incoming[c.out_id].append(c)
                indeg[c.out_id] += 1
        order: List[int] = []
        S = [nid for nid in self.nodes if self.nodes[nid].type==INPUT]
        S += [nid for nid in self.nodes if indeg[nid]==0 and nid not in S]
        seen = set()
        while S:
            nid = S.pop()
            if nid in seen: continue
            seen.add(nid)
            order.append(nid)
            for c in outgoing[nid]:
                indeg[c.out_id] -= 1
                if indeg[c.out_id]==0:
                    S.append(c.out_id)
        if len(order) < len(self.nodes):
            # 循環検知 -> ランダムで数本無効化して回避
            for _ in range(3):
                if not self.conns: break
                rand.choice(list(self.conns.values())).enabled = False
            self.topo_cache_valid = False
            self._ensure_topo()
            return
        self.topo_order = order
        self._incoming = incoming
        self._topo_pos = {nid:i for i,nid in enumerate(order)}
        self.topo_cache_valid = True

    # ---------- シリアライズ ----------
    def to_dict(self) -> dict:
        return {
            "nodes": [asdict(n) for n in self.nodes.values()],
            "conns": [asdict(c) for c in self.conns.values()],
            "in_ids": self.in_ids,
            "out_ids": self.out_ids
        }

    @staticmethod
    def from_dict(d: dict) -> 'Genome':
        g = Genome(d["in_ids"], d["out_ids"])
        g.nodes = {n["id"]: NodeGene(**n) for n in d["nodes"]}
        g.conns = {c["innov"]: ConnGene(**c) for c in d["conns"]}
        g.topo_cache_valid = False
        return g

# ===================== 入出力ノード定義 =====================
N_RAYS = 5
IN_FEATURES = N_RAYS*3 + 4
OUT_FEATURES = 5
INPUT_IDS  = list(range(0, IN_FEATURES))
OUTPUT_IDS = list(range(100, 100+OUT_FEATURES))

def brain_init_genome() -> Genome:
    g = Genome(INPUT_IDS, OUTPUT_IDS)
    for i in INPUT_IDS:
        for o in OUTPUT_IDS:
            if rand.random() < 0.2:
                innov = INNOV_DB.get_innov(i, o)
                g.conns[innov] = ConnGene(innov, i, o, rng.normal(0,1.0), True)
    if not g.conns:
        i = rand.choice(INPUT_IDS); o = rand.choice(OUTPUT_IDS)
        innov = INNOV_DB.get_innov(i, o)
        g.conns[innov] = ConnGene(innov, i, o, rng.normal(0,1.0), True)
    g.topo_cache_valid = False
    return g

# ===================== 個体と世界 =====================
@dataclass
class Food:
    x: float
    y: float
    e: float

@dataclass
class Body:
    x: float
    y: float
    e: float

class Agent:
    __slots__ = ("x","y","vx","vy","S","E","brain","base_color","eaten_units")
    def __init__(self, genome: Optional[Genome]=None):
        self.x = rng.uniform(0, W)
        self.y = rng.uniform(0, H)
        ang = rng.uniform(0, 2*np.pi)
        spd = rng.uniform(0, 1.0)
        self.vx, self.vy = spd*math.cos(ang), spd*math.sin(ang)
        self.S = rng.uniform(0.8, 1.2)
        self.E = 200.0
        self.brain: Genome = genome if genome is not None else brain_init_genome()
        self.base_color = self._color_from_genome()
        self.eaten_units = 0.0

    def _color_from_genome(self):
        key = 0
        for k in sorted(self.brain.conns.keys()):
            c = self.brain.conns[k]
            key = (key*1315423911 + k + int(abs(c.w)*1000)) & 0xFFFFFFFF
        h = (key % 360) / 360.0
        return hsl_to_rgb(h, 0.65, 0.5)

    def sense(self, neighbors: List['Agent']) -> Dict[int,float]:
        angles = np.linspace(-math.radians(60), math.radians(60), N_RAYS)
        theta  = math.atan2(self.vy, self.vx + 1e-9)
        feats: List[float] = []
        for a in angles:
            dirx, diry = math.cos(theta+a), math.sin(theta+a)
            best_d = R_SENSE
            best_S = 0.0
            best_vproj = 0.0
            for other in neighbors:
                if other is self: continue
                dx = torus_delta(other.x - self.x, W)
                dy = torus_delta(other.y - self.y, H)
                d = math.hypot(dx, dy)
                if d < best_d and d > 1e-6:
                    if (dx*dirx + dy*diry)/max(d,1e-6) > math.cos(math.radians(90)):
                        best_d = d
                        best_S = other.S / self.S
                        relv = ((other.vx - self.vx)*dirx + (other.vy - self.vy)*diry)
                        best_vproj = relv
            feats += [best_d/R_SENSE, best_S, math.tanh(best_vproj/5.0)]
        spd = math.hypot(self.vx, self.vy)/max(1e-6, SPEED_MAX_BASE)
        feats += [self.E/400.0, self.S/1.5, spd, 0.0]
        x = {nid:0.0 for nid in INPUT_IDS}
        for i,v in enumerate(feats[:len(INPUT_IDS)]):
            x[INPUT_IDS[i]] = v
        return x

    def step(self, neighbors: List['Agent']) -> Tuple[bool,bool,float,float,float]:
        x = self.sense(neighbors)
        o = self.brain.forward(x)
        thrust = math.tanh(o[OUTPUT_IDS[0]])
        turn   = math.tanh(o[OUTPUT_IDS[1]]) * 0.3
        attack = o[OUTPUT_IDS[2]] > 0.5
        mate   = o[OUTPUT_IDS[3]] > 0.5
        eat_strength = float(max(0.0, min(1.0, o[OUTPUT_IDS[4]])))

        theta = math.atan2(self.vy, self.vx + 1e-9) + turn
        vmax  = SPEED_MAX_BASE*(1.2 - 0.2*self.S)
        spd   = np.clip(math.hypot(self.vx, self.vy) + thrust, 0, vmax)
        self.vx, self.vy = spd*math.cos(theta), spd*math.sin(theta)
        self.x = wrap(self.x + self.vx*DT, W)
        self.y = wrap(self.y + self.vy*DT, H)

        # move_cost  = MOVE_COST_K * spd*spd
        # brain_cost = BRAIN_COST_PER_CONN * len(self.brain.conns)
        # self.E -= (move_cost + brain_cost)
        # ---- 新しい代謝：基礎代謝 + 飢餓 + アイドル・ペナルティ ----
        move_cost  = MOVE_COST_K * spd * spd
        brain_cost = BRAIN_COST_PER_CONN * len(self.brain.conns)

        base_cost  = BASE_COST

        # 飢餓域での追加ドレイン（Eが閾値を下回ると常時発生）
        starvation_cost = STARVATION_COST if self.E < STARVATION_E else 0.0
        # さらに「低速＆出力小」のときはアイドル・ペナルティ
        is_idle_speed = (spd < IDLE_SPEED_FRAC * vmax)
        is_idle_cmd   = (abs(thrust) < IDLE_THRUST_TH and abs(turn) < IDLE_TURN_TH)
        idle_cost = IDLE_COST if (is_idle_speed and is_idle_cmd) else 0.0

        self.E -= (move_cost + brain_cost + base_cost + starvation_cost + idle_cost)

        return attack, mate, eat_strength, spd, vmax

class World:
    def __init__(self, from_last10: bool=False):
        self.t = 0
        self.agents: List[Agent] = []
        self.foods: List[Food] = []
        self.bodies: List[Body] = []
        self.births = 0
        self.deaths = 0
        self.grid: List[List[List[int]]] = [[[] for _ in range(GRID_H)] for __ in range(GRID_W)]

        # 初期フードのばら撒き（足りない初期餌問題の対策）
        if INITIAL_FOOD_SCALE:
            # 面積 2000x2000 を基準にスケール
            area_scale = (W * H) / (2000.0 * 2000.0)
            n_seed = int(INITIAL_FOOD_PIECES * area_scale)
        else:
            n_seed = int(INITIAL_FOOD_PIECES)
        self._seed_food(n_seed)

        if from_last10 and os.path.exists(LAST10_PATH):
            data = self._load_last10()
            if data:
                self._bootstrap_from_last10_diverse(data, N_INIT)

        if not self.agents:
            self.agents = [Agent() for _ in range(N_INIT)]

    # ---------- グリッド ----------
    def reset_grid(self):
        for x in range(GRID_W):
            for y in range(GRID_H):
                self.grid[x][y].clear()

    def insert_grid(self):
        for i,a in enumerate(self.agents):
            gx = int(a.x // CELL) % GRID_W
            gy = int(a.y // CELL) % GRID_H
            self.grid[gx][gy].append(i)

    def neighbors(self, a: Agent, radius=R_SENSE) -> List[Agent]:
        cx = int(a.x // CELL)
        cy = int(a.y // CELL)
        rcell = max(1, int(math.ceil(radius / CELL)))
        res_idx = []
        for dx in range(-rcell, rcell+1):
            for dy in range(-rcell, rcell+1):
                gx = (cx + dx) % GRID_W
                gy = (cy + dy) % GRID_H
                res_idx.extend(self.grid[gx][gy])
        out = []
        for idx in res_idx:
            b = self.agents[idx]
            dx = torus_delta(b.x - a.x, W)
            dy = torus_delta(b.y - a.y, H)
            if dx*dx + dy*dy <= radius*radius:
                out.append(b)
        return out

    # ---------- 生態 ----------
    def spawn_food(self):
        # Note: ランタイムのフード出現密度は FOOD_RATE で制御します。増減で継続的な餌の量を調整可能。
        if rand.random() < FOOD_RATE:
            self.foods.append(Food(rand.uniform(0,W), rand.uniform(0,H), FOOD_EN))

    def _seed_food(self, n: int):
        """起動直後にフードを一括投入して初期飢餓を防ぐ。"""
        for _ in range(max(0, int(n))):
            self.foods.append(Food(rand.uniform(0, W), rand.uniform(0, H), FOOD_EN))

    def is_compatible(self, a: Agent, b: Agent) -> bool:
        d_beh = ( (a.vx-b.vx)**2 + (a.vy-b.vy)**2 )**0.5 / SPEED_MAX_BASE
        sa = set(a.brain.conns.keys()); sb = set(b.brain.conns.keys())
        d_gen = len(sa ^ sb) / (len(sa | sb)+1e-6)
        return ASSORT_ALPHA*d_gen + (1-ASSORT_ALPHA)*d_beh < COMP_TH

    def tick(self, substeps=1):
        births_this = 0; deaths_this = 0
        for _ in range(substeps):
            self.t += 1
            self.spawn_food()
            self.reset_grid()
            self.insert_grid()

            intents = [a.step(self.neighbors(a, R_SENSE)) for a in self.agents]

            # 食餌・死体スカベンジ
            for i,a in enumerate(self.agents):
                if a.E <= 0: continue
                attack, mate, eat_str, spd, vmax = intents[i]
                # 食餌
                for f in self.foods:
                    dx = torus_delta(f.x - a.x, W); dy = torus_delta(f.y - a.y, H)
                    if dx*dx + dy*dy < R_HIT*R_HIT and f.e > 0:
                        take = min(f.e, FOOD_EN) * (0.3 + 0.7*eat_str)
                        a.E += take; f.e -= take
                        a.eaten_units += take / FOOD_EN
                # 死体
                for bdy in self.bodies:
                    dx = torus_delta(bdy.x - a.x, W); dy = torus_delta(bdy.y - a.y, H)
                    if dx*dx + dy*dy < R_HIT*R_HIT and bdy.e > 0:
                        take = min(bdy.e, 10.0*(0.3 + 0.7*eat_str))
                        a.E += take; bdy.e -= take
                        a.eaten_units += take / FOOD_EN

            # 近接相互作用（交配/攻撃/分裂）
            newborns: List[Agent] = []
            removed = set()

            # 交配・攻撃
            for i,a in enumerate(self.agents):
                if i in removed or a.E <= 0: continue
                attack_i, mate_i, eat_i, _, _ = intents[i]
                neigh = self.neighbors(a, R_HIT*1.2)
                for b in neigh:
                    if b is a: continue
                    j = self.agents.index(b)
                    if j in removed or b.E <= 0: continue
                    dx = torus_delta(b.x - a.x, W); dy = torus_delta(b.y - a.y, H)
                    if dx*dx + dy*dy < R_HIT*R_HIT:
                        # 交配
                        if mate_i and intents[j][1]:
                            if a.E > E_BIRTH_THRESHOLD and b.E > E_BIRTH_THRESHOLD and self.is_compatible(a,b):
                                child_g = Genome.crossover(a.brain, b.brain)
                                child_g.structural_mutation()
                                child = Agent(child_g); child.x, child.y = a.x, a.y; child.E = CHILD_EN
                                a.E -= PARENT_COST; b.E -= PARENT_COST
                                newborns.append(child); births_this += 1
                        # 攻撃
                        if attack_i:
                            atk = 8.0*a.S; dfn = 5.0*b.S
                            if atk > dfn:
                                a.E += 60.0; removed.add(j)
                                self.bodies.append(Body(b.x, b.y, BODY_INIT_EN)); deaths_this += 1

            # 分裂（無性）
            for i,a in enumerate(self.agents):
                if i in removed or a.E <= 0: continue
                if a.E > FISSION_ENERGY_TH and a.eaten_units >= FISSION_FOOD_UNITS_TH:
                    child_g = a.brain.clone().micro_mutate(weight_sigma=0.03, p_add_conn=0.04, p_add_node=0.015, p_del_conn=0.008)
                    child = Agent(child_g); child.x, child.y = a.x, a.y; child.E = FISSION_CHILD_EN
                    a.E -= FISSION_PARENT_COST; a.eaten_units = 0.0
                    newborns.append(child); births_this += 1

            # 片付け
            keep = []
            for k,ag in enumerate(self.agents):
                dead = (k in removed) or (ag.E <= 0)
                if dead:
                    self.bodies.append(Body(ag.x, ag.y, BODY_INIT_EN*(1.0 if k in removed else 0.5)))
                    deaths_this += 1
                else:
                    keep.append(ag)
            self.agents = keep

            # 死体減衰・餌整理
            for b in self.bodies: b.e *= DECAY_BODY
            self.bodies = [b for b in self.bodies if b.e > 1.0]
            self.foods  = [f for f in self.foods if f.e > 1.0]

            if newborns:
                self.agents.extend(newborns)

            # 絶滅時：last10 からクラスタ均等サンプルで再播種
            if not self.agents:
                data = self._load_last10()
                if data:
                    print(f"[extinct] reseed from last10 (diverse) -> {N_INIT}")
                    self._bootstrap_from_last10_diverse(data, N_INIT)
                else:
                    print("[extinct] no last10 -> random reinit")
                    self.agents = [Agent() for _ in range(N_INIT)]

        self.births += births_this; self.deaths += deaths_this

    # ---------- 最後の10個体 保存/復元 ----------
    def snapshot_topK(self) -> List[dict]:
        top = sorted(self.agents, key=lambda a: a.E, reverse=True)[:SAVE_TOP_K]
        pack = []
        for a in top:
            pack.append({"genome": a.brain.to_dict(), "S": a.S})
        return pack

    def save_last10(self):
        data = self.snapshot_topK()
        if data:
            with open(LAST10_PATH, "w") as f:
                json.dump(data, f)
            print(f"[saved last {len(data)}] -> {LAST10_PATH}")

    def _load_last10(self) -> list:
        try:
            with open(LAST10_PATH, "r") as f:
                data = json.load(f)
        except Exception:
            data = []
        return data

    # ---------- 多様性ブートストラップ ----------
    def _bootstrap_from_last10_diverse(self, data: list, n_target: int) -> None:
        if not data:
            self.agents = []
            return

        # --- 特徴ベクトル化 ---
        X = []
        for item in data:
            g = Genome.from_dict(item["genome"])
            X.append(self._genome_features(g))
        X = np.stack(X, axis=0)  # shape: (n, d)
        n = X.shape[0]

        # すべて同一（SSE ~ 0）ならクラスタリングはスキップし、均等ローテで増殖
        all_same = np.allclose(X, X[0], atol=1e-8)

        # --- k の安定化 ---
        if n == 1:
            # クラスタリング不要。1個体ベースで n_target まで微小変異増殖
            self.agents = []
            base_item = data[0]
            for _ in range(n_target):
                g_child = Genome.from_dict(base_item["genome"]).micro_mutate(
                    weight_sigma=0.03, p_add_conn=0.04, p_add_node=0.015, p_del_conn=0.008
                )
                a = Agent(g_child)
                a.S = float(base_item.get("S", a.S)) * float(np.clip(rng.normal(1.0, 0.03), 0.9, 1.1))
                a.E = 220.0
                self.agents.append(a)
            print("[diverse bootstrap] n=1 -> simple replicate")
            return

        # 通常は 2..min(KMEANS_MAX_K, n) の範囲
        k_max = min(KMEANS_MAX_K, n)
        k = max(2, k_max) if n >= 4 else min(2, k_max)  # n=2→k=2, n=3→k=2, n>=4→k=min(KMAX,n)

        if all_same:
            # すべて同一点：ラベルを均等に割り振るだけ
            labels = np.arange(n) % k
            clusters = {i: list(np.where(labels == i)[0]) for i in range(k)}
        else:
            labels, centers, sse = self._kmeans_best_of_n(X, k, KMEANS_TRIES, KMEANS_ITERS)
            clusters = {i: [] for i in range(k)}
            for idx, lab in enumerate(labels):
                clusters[int(lab)].append(idx)

        # --- ラウンドロビン均等サンプリングで n_target 体を生成 ---
        self.agents = []
        order = list(range(len(clusters)))
        ptr = {i: 0 for i in order}
        # 万一、空クラスタしか残らない場合に備えて平滑化
        non_empty = [i for i in order if clusters[i]]
        if not non_empty:
            # 全部空（理論上起きにくい）→全データローテで生成
            non_empty = [0]
            clusters[0] = list(range(n))

        while len(self.agents) < n_target:
            progressed = False
            for c in non_empty:
                if len(self.agents) >= n_target:
                    break
                if not clusters[c]:
                    continue
                idx = clusters[c][ptr[c] % len(clusters[c])]
                ptr[c] += 1
                item = data[idx]
                g_child = Genome.from_dict(item["genome"]).micro_mutate(
                    weight_sigma=0.03, p_add_conn=0.04, p_add_node=0.015, p_del_conn=0.008
                )
                a = Agent(g_child)
                a.S = float(item.get("S", a.S)) * float(np.clip(rng.normal(1.0, 0.03), 0.9, 1.1))
                a.E = 220.0
                self.agents.append(a)
                progressed = True
            if not progressed:
                # 保険：何も進まない場合はランダム増殖
                self.agents.append(Agent())

        print(f"[diverse bootstrap] k={k} clusters sizes={[len(clusters[i]) for i in range(len(clusters))]}")

    # ---------- ゲノム→特徴量 ----------
    def _genome_features(self, g: Genome) -> np.ndarray:
        """
        特徴ハッシュ（KMEANS_DIM）:
          - 各結合 (innov, weight) -> index = hash(innov) % KMEANS_DIM に |w| を加算
          - 追加統計: [num_nodes, num_conns, num_hidden, mean|w|, std|w|, mean|bias|]
        """
        feat = np.zeros(KMEANS_DIM + 6, dtype=np.float32)
        # hash-bucket
        for innov, c in g.conns.items():
            idx = (innov * 2654435761) & 0xFFFFFFFF
            feat[idx % KMEANS_DIM] += abs(c.w)
        # stats
        num_nodes = len(g.nodes)
        num_conns = len(g.conns)
        num_hidden = sum(1 for n in g.nodes.values() if n.type == HIDDEN)
        absw = [abs(c.w) for c in g.conns.values()] or [0.0]
        absb = [abs(n.bias) for n in g.nodes.values()] or [0.0]
        feat[KMEANS_DIM + 0] = num_nodes
        feat[KMEANS_DIM + 1] = num_conns
        feat[KMEANS_DIM + 2] = num_hidden
        feat[KMEANS_DIM + 3] = float(np.mean(absw))
        feat[KMEANS_DIM + 4] = float(np.std(absw))
        feat[KMEANS_DIM + 5] = float(np.mean(absb))
        # 正規化（L2）
        # norm = np.linalg.norm(feat) + 1e-8
        # 正規化（L2）
        norm = float(np.linalg.norm(feat))
        if not np.isfinite(norm) or norm < 1e-12:
            return feat  # そのまま返す（全ゼロは実質同一点扱い）
        return feat / norm
        return feat / norm

    # ---------- 簡易 K-means ----------
    def _kmeans(self, X: np.ndarray, k: int, iters: int = 50) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        K-means++ 初期化を堅牢化：
          - 確率分布がゼロ和/NaN の場合は一様分布へフォールバック
          - 同一点群（dist合計 ~ 0）の場合は重複しないランダム初期化へ
          - 空クラスタはランダム再初期化
        """
        n, d = X.shape
        # k が n を超えないように安全化
        k = min(k, n)
        # すべて同一ベクトル？
        if np.allclose(X, X[0], atol=1e-8):
            centers = X[np.random.choice(n, size=k, replace=False)]
        else:
            centers = np.zeros((k, d), dtype=np.float32)
            chosen = set()
            idx0 = np.random.randint(0, n)
            centers[0] = X[idx0]
            chosen.add(idx0)
            dist = np.full(n, np.inf, dtype=np.float64)
            for i in range(1, k):
                # 直近センタとの距離最小を更新
                d2_new = np.sum((X - centers[i-1])**2, axis=1)
                dist = np.minimum(dist, d2_new)
                total = np.sum(dist)
                if not np.isfinite(total) or total <= 1e-12:
                    # ほぼ同一点 or 数値不良 → 未選択点から均等ランダム
                    candidates = [j for j in range(n) if j not in chosen]
                    if not candidates:
                        candidates = list(range(n))
                    idx = random.choice(candidates)
                else:
                    probs = dist / total
                    # 数値安定化
                    probs = np.clip(probs, 0.0, 1.0)
                    s = probs.sum()
                    if s <= 0.0 or not np.isfinite(s):
                        probs = np.full(n, 1.0/n)
                    else:
                        probs = probs / s
                    idx = np.random.choice(n, p=probs)
                centers[i] = X[idx]
                chosen.add(idx)

        labels = np.zeros(n, dtype=np.int32)
        for _ in range(iters):
            # assign
            d2 = ((X[:, None, :] - centers[None, :, :])**2).sum(axis=2)
            labels = np.argmin(d2, axis=1)

            # update
            new_centers = centers.copy()
            for i in range(k):
                mask = (labels == i)
                if np.any(mask):
                    new_centers[i] = X[mask].mean(axis=0)
                else:
                    # 空クラスタ → ランダム再初期化（既存点から）
                    new_centers[i] = X[np.random.randint(0, n)]
            # 収束判定
            if np.allclose(new_centers, centers, atol=1e-5):
                centers = new_centers
                break
            centers = new_centers

        sse = float(((X - centers[labels])**2).sum())
        return labels, centers, sse

    def _kmeans_best_of_n(self, X: np.ndarray, k: int, tries: int, iters: int) -> Tuple[np.ndarray, np.ndarray, float]:
        best = None
        for _ in range(tries):
            labels, centers, sse = self._kmeans(X, k, iters)
            if (best is None) or (sse < best[2]):
                best = (labels, centers, sse)
        return best

    # ---------- I/O ----------
    def _load_last10_or_random(self):
        data = self._load_last10()
        if data:
            print(f"[reseed] bootstrap from last10 (diverse) -> {N_INIT}")
            self._bootstrap_from_last10_diverse(data, N_INIT)
        else:
            print("[reset] no last10 available -> random reinit")
            self.agents = [Agent() for _ in range(N_INIT)]


__all__ = [
    "Agent",
    "World",
    "Genome",
    "InnovationDB",
    "LAST10_PATH",
    "N_INIT",
    "DT",
]
