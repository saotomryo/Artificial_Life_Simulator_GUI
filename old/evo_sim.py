# evo_sim.py
import os
import math
import time
import random
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

# ----------- 依存 -----------
# pip install pygame
import pygame

# ============ 物理/生態パラメータ ============
W, H = 2000.0, 2000.0              # ワールドサイズ
N_INIT = 200                       # 初期個体数
DT = 0.15                          # シミュレーション刻み
SPEED_MAX_BASE = 8.0
VISION_DEG = 180.0
R_SENSE = 180.0
R_HIT = 12.0

FOOD_RATE = 0.015                  # 1tickあたり環境餌スポーン確率
FOOD_EN = 50.0
DECAY_BODY = 0.995                 # 死体のエネルギー減衰
BODY_INIT_EN = 80.0

MOVE_COST_K = 0.002
BRAIN_COST_K = 0.0002

# NN 形状（MVPは固定トポロジ）
IN_DIM = 5*3 + 4
H_DIM = 16
OUT_DIM = 5

# 交配条件
E_BIRTH_THRESHOLD = 220.0
CHILD_EN = 120.0
PARENT_COST = 80.0

# 変異
MUT_SIGMA = 0.08

# 空間分割（セルグリッド）
CELL = 40.0  # セル辺長（R_SENSE以上は不要、近傍探索用）
GRID_W = int(math.ceil(W / CELL))
GRID_H = int(math.ceil(H / CELL))

rng = np.random.default_rng(0)
rand = random.Random(0)

# ============ ユーティリティ ============
def wrap(v, L):
    if v < 0: return v + L
    if v >= L: return v - L
    return v

def torus_delta(dx, L):
    # 最短トーラス差分
    if dx >  L/2: dx -= L
    if dx < -L/2: dx += L
    return dx

def param_count(p): return sum(v.size for v in p.values())

def flat_params(p): return np.concatenate([p["W1"].ravel(), p["b1"], p["W2"].ravel(), p["b2"]])

def mlp_init(in_dim, h_dim, out_dim):
    return {
        "W1": rng.normal(0, 0.5, (h_dim, in_dim)),
        "b1": np.zeros(h_dim),
        "W2": rng.normal(0, 0.5, (out_dim, h_dim)),
        "b2": np.zeros(out_dim),
    }

def mlp_forward(p, x):
    h = np.tanh(p["W1"] @ x + p["b1"])
    y = p["W2"] @ h + p["b2"]
    return y

def mutate(p, sigma=MUT_SIGMA):
    q = {k: v.copy() for k,v in p.items()}
    for k in q:
        q[k] += rng.normal(0, sigma, q[k].shape)
    return q

def crossover(pa, pb):
    qa = {k: v.copy() for k,v in pa.items()}
    for k in qa:
        mask = rng.random(qa[k].shape) < 0.5
        qa[k][mask] = pb[k][mask]
    return qa

def is_compatible(a, b, alpha=0.5, th=1.5):
    d_w = np.linalg.norm(flat_params(a.brain) - flat_params(b.brain)) / 200.0
    d_beh = np.linalg.norm(np.array([a.vx,a.vy]) - np.array([b.vx,b.vy]))/SPEED_MAX_BASE
    return alpha*d_w + (1-alpha)*d_beh < th

def hash_color_from_params(p) -> Tuple[int,int,int]:
    # パラメータベクトルのハッシュからH(色相)を決定
    v = flat_params(p)
    h = (int(abs(hash(v.tobytes())) % 360)) / 360.0
    s = 0.65
    l = 0.5
    return hsl_to_rgb(h, s, l)

def hsl_to_rgb(h, s, l):
    # h[0,1) s[0,1] l[0,1] -> 0..255 RGB
    import colorsys
    r,g,b = colorsys.hls_to_rgb(h, l, s)
    return int(r*255), int(g*255), int(b*255)

# ============ 個体と環境 ============

class Agent:
    __slots__ = ("x","y","vx","vy","S","E","brain","base_color")
    def __init__(self):
        self.x = rng.uniform(0, W)
        self.y = rng.uniform(0, H)
        ang = rng.uniform(0, 2*np.pi)
        spd = rng.uniform(0, 1.0)
        self.vx, self.vy = spd*np.cos(ang), spd*np.sin(ang)
        self.S = rng.uniform(0.8, 1.2)
        self.E = 200.0
        self.brain = mlp_init(IN_DIM, H_DIM, OUT_DIM)
        self.base_color = hash_color_from_params(self.brain)

    def sense(self, neighbors):
        # 前方±60°の5レイ、最短ターゲットの特徴
        angles = np.linspace(-np.deg2rad(60), np.deg2rad(60), 5)
        theta = math.atan2(self.vy, self.vx + 1e-9)
        feats = []
        dirf = np.array([math.cos(theta), math.sin(theta)])
        for a in angles:
            dirx, diry = math.cos(theta+a), math.sin(theta+a)
            best_d = R_SENSE
            best_S = 0.0
            best_vproj = 0.0
            # 近傍に限定
            for other in neighbors:
                if other is self: continue
                dx = torus_delta(other.x - self.x, W)
                dy = torus_delta(other.y - self.y, H)
                d = math.hypot(dx, dy)
                if d < best_d and d > 1e-6:
                    dir2 = (dx/d, dy/d)
                    if dir2[0]*dirx + dir2[1]*diry > math.cos(math.radians(90)):
                        best_d = d
                        best_S = other.S / self.S
                        relv = ((other.vx - self.vx)*dirx + (other.vy - self.vy)*diry)
                        best_vproj = relv
            feats += [best_d/R_SENSE, best_S, math.tanh(best_vproj/5.0)]

        spd = math.hypot(self.vx, self.vy)/max(1e-6, SPEED_MAX_BASE)
        feats += [self.E/400.0, self.S/1.5, spd, 0.0]
        return np.array(feats, dtype=np.float32)

    def step(self, neighbors):
        x = self.sense(neighbors)
        o = mlp_forward(self.brain, x)
        thrust = math.tanh(o[0])
        turn = math.tanh(o[1]) * 0.3
        attack = o[2] > 0.5
        mate   = o[3] > 0.5
        eat_strength = float(np.clip(o[4], 0, 1))

        theta = math.atan2(self.vy, self.vx + 1e-9) + turn
        vmax = SPEED_MAX_BASE*(1.2 - 0.2*self.S)
        spd = np.clip(math.hypot(self.vx, self.vy) + thrust, 0, vmax)
        self.vx, self.vy = spd*math.cos(theta), spd*math.sin(theta)
        self.x = wrap(self.x + self.vx*DT, W)
        self.y = wrap(self.y + self.vy*DT, H)

        move_cost  = MOVE_COST_K * spd*spd
        brain_cost = BRAIN_COST_K * param_count(self.brain)
        self.E -= (move_cost + brain_cost)

        return attack, mate, eat_strength

# 環境オブジェクト
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

class World:
    def __init__(self):
        self.t = 0
        self.agents: List[Agent] = [Agent() for _ in range(N_INIT)]
        self.foods: List[Food] = []
        self.bodies: List[Body] = []
        self.births = 0
        self.deaths = 0

        # 空間グリッド（インデックス: (gx,gy) -> agent idx list）
        self.grid: List[List[List[int]]] = [[[] for _ in range(GRID_H)] for __ in range(GRID_W)]

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
        # 半径radius内の近傍候補（周辺セル探索）
        cx = int(a.x // CELL)
        cy = int(a.y // CELL)
        rcell = max(1, int(math.ceil(radius / CELL)))
        res_idx = []
        for dx in range(-rcell, rcell+1):
            for dy in range(-rcell, rcell+1):
                gx = (cx + dx) % GRID_W
                gy = (cy + dy) % GRID_H
                res_idx.extend(self.grid[gx][gy])
        # フィルタリング（実距離チェック）
        out = []
        for idx in res_idx:
            b = self.agents[idx]
            if b is a: 
                out.append(b)  # senseでは自分は無視しているが、stepの引数合わせで含める
                continue
            dx = torus_delta(b.x - a.x, W)
            dy = torus_delta(b.y - a.y, H)
            if dx*dx + dy*dy <= radius*radius:
                out.append(b)
        return out

    def spawn_food(self):
        if rand.random() < FOOD_RATE:
            self.foods.append(Food(rand.uniform(0,W), rand.uniform(0,H), FOOD_EN))

    def tick(self, substeps=1):
        births_this = 0
        deaths_this = 0
        for _ in range(substeps):
            self.t += 1
            self.spawn_food()
            # 空間グリッド更新
            self.reset_grid()
            self.insert_grid()

            # 行動
            intents = []
            for a in self.agents:
                intents.append(a.step(self.neighbors(a, R_SENSE)))

            # 相互作用
            newborns: List[Agent] = []
            removed = set()

            # 食べ物・死体（各agent）
            for i,a in enumerate(self.agents):
                if i in removed or a.E <= 0: 
                    continue
                attack, mate, eat_str = intents[i]

                # 食べ物
                for f in self.foods:
                    dx = torus_delta(f.x - a.x, W)
                    dy = torus_delta(f.y - a.y, H)
                    if dx*dx + dy*dy < R_HIT*R_HIT and f.e > 0:
                        take = f.e * (0.3 + 0.7*eat_str)
                        a.E += take
                        f.e -= take

                # 死体
                for bdy in self.bodies:
                    dx = torus_delta(bdy.x - a.x, W)
                    dy = torus_delta(bdy.y - a.y, H)
                    if dx*dx + dy*dy < R_HIT*R_HIT and bdy.e > 0:
                        take = min(bdy.e, 10.0*(0.3 + 0.7*eat_str))
                        a.E += take
                        bdy.e -= take

            # 近接個体どうし（周辺セルのみ）
            for i,a in enumerate(self.agents):
                if i in removed or a.E <= 0: 
                    continue
                attack_i, mate_i, eat_i = intents[i]
                neigh = self.neighbors(a, R_HIT*1.2)
                for b in neigh:
                    if b is a: 
                        continue
                    j = self.agents.index(b)  # 近傍数が少ないので許容。最適化するならID管理
                    if j in removed or b.E <= 0: 
                        continue
                    dx = torus_delta(b.x - a.x, W)
                    dy = torus_delta(b.y - a.y, H)
                    if dx*dx + dy*dy < R_HIT*R_HIT:
                        # 交配
                        if mate_i and intents[j][1]:
                            if a.E > E_BIRTH_THRESHOLD and b.E > E_BIRTH_THRESHOLD and is_compatible(a,b):
                                child = Agent()
                                child.brain = crossover(a.brain, b.brain)
                                child.brain = mutate(child.brain, sigma=MUT_SIGMA)
                                child.base_color = hash_color_from_params(child.brain)
                                child.x, child.y = a.x, a.y
                                child.E = CHILD_EN
                                a.E -= PARENT_COST
                                b.E -= PARENT_COST
                                newborns.append(child)
                                births_this += 1
                        # 攻撃
                        if attack_i:
                            atk = 8.0*a.S
                            dfn = 5.0*b.S
                            if atk > dfn:
                                a.E += 60.0
                                removed.add(j)
                                self.bodies.append(Body(b.x, b.y, BODY_INIT_EN))
                                deaths_this += 1

            # 片付け
            if removed:
                keep = []
                for k,ag in enumerate(self.agents):
                    if k in removed or ag.E <= 0:
                        if k not in removed and ag.E <= 0:
                            self.bodies.append(Body(ag.x, ag.y, BODY_INIT_EN * 0.5))
                            deaths_this += 1
                        continue
                    keep.append(ag)
                self.agents = keep
            else:
                # エネルギー枯渇死
                keep = []
                for k,ag in enumerate(self.agents):
                    if ag.E <= 0:
                        self.bodies.append(Body(ag.x, ag.y, BODY_INIT_EN * 0.5))
                        deaths_this += 1
                        continue
                    keep.append(ag)
                self.agents = keep

            # 死体減衰・餌整理
            for b in self.bodies:
                b.e *= DECAY_BODY
            self.bodies = [b for b in self.bodies if b.e > 1.0]
            self.foods  = [f for f in self.foods if f.e > 1.0]

            # 出生反映
            if newborns:
                self.agents.extend(newborns)

            # 安全弁：絶滅時はリセット
            if not self.agents:
                self.agents = [Agent() for _ in range(N_INIT)]

        self.births += births_this
        self.deaths += deaths_this
        return births_this, deaths_this

# ============ 可視化（Pygame） ============
class Viewer:
    def __init__(self, world: World, win_w=1280, win_h=800):
        pygame.init()
        pygame.display.set_caption("Artificial Life Simulator")
        self.world = world
        self.win_w, self.win_h = win_w, win_h
        self.screen = pygame.display.set_mode((win_w, win_h))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas, Menlo, Monospace", 16)
        self.bg = (18,18,22)

        # スケール・オフセット（ワールド→画面）
        self.scale = min(win_w/W, win_h/H)
        self.offset = (0,0)  # 今は原点合わせ

        # 実行制御
        self.paused = False
        self.fast = False
        self.substeps_normal = 1
        self.substeps_fast = 6

        os.makedirs("screens", exist_ok=True)

    def world_to_screen(self, x, y):
        sx = int(x * self.scale) + self.offset[0]
        sy = int(y * self.scale) + self.offset[1]
        return sx, sy

    def draw(self):
        self.screen.fill(self.bg)
        # 食べ物・死体
        for f in self.world.foods:
            sx, sy = self.world_to_screen(f.x, f.y)
            r = max(1, int(2 * self.scale))
            pygame.draw.circle(self.screen, (40, 200, 60), (sx, sy), r)

        for b in self.world.bodies:
            sx, sy = self.world_to_screen(b.x, b.y)
            r = max(1, int(3 * self.scale))
            pygame.draw.circle(self.screen, (140, 100, 60), (sx, sy), r)

        # 個体
        for a in self.world.agents:
            sx, sy = self.world_to_screen(a.x, a.y)
            # 体格とエネルギーで見た目反映
            r = max(2, int((3.0 + 2.0*(a.S-0.8)/0.4) * self.scale))
            # エネルギーで明度調整
            e_norm = max(0.1, min(1.0, a.E/300.0))
            col = tuple(min(255, int(c*e_norm + 20)) for c in a.base_color)
            pygame.draw.circle(self.screen, col, (sx, sy), r)

            # 向きライン（短め）
            ang = math.atan2(a.vy, a.vx + 1e-9)
            lx = int(sx + math.cos(ang) * 6)
            ly = int(sy + math.sin(ang) * 6)
            pygame.draw.line(self.screen, (230,230,230), (sx,sy), (lx,ly), 1)

        # HUD
        n = len(self.world.agents)
        meanE = (sum(a.E for a in self.world.agents)/n) if n>0 else 0.0
        txt = (
            f"t={self.world.t} | n={n} | meanE={meanE:5.1f} | "
            f"births={self.world.births} deaths={self.world.deaths} | "
            f"{'FAST' if self.fast else 'NORMAL'}"
        )
        fps_txt = f"FPS={self.clock.get_fps():.1f}"
        surf = self.font.render(txt, True, (220,220,220))
        self.screen.blit(surf, (8, 8))
        surf2 = self.font.render(fps_txt, True, (180,180,180))
        self.screen.blit(surf2, (8, 28))

        pygame.display.flip()

    def run(self):
        running = True
        while running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN:
                    if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif ev.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif ev.key == pygame.K_f:
                        self.fast = not self.fast
                    elif ev.key == pygame.K_s:
                        fname = f"screens/snap_{int(time.time())}.png"
                        pygame.image.save(self.screen, fname)
                        print(f"[saved] {fname}")
                    elif ev.key == pygame.K_r:
                        self.world = World()

            substeps = 0 if self.paused else (self.substeps_fast if self.fast else self.substeps_normal)
            if substeps > 0:
                self.world.tick(substeps=substeps)

            self.draw()
            # 表示を見やすくするためFPS上限
            self.clock.tick(60)

        pygame.quit()

# ============ エントリ ============

def main():
    world = World()
    viewer = Viewer(world, win_w=1280, win_h=800)
    viewer.run()

if __name__ == "__main__":
    main()