import numpy as np

W, H = 2000.0, 2000.0
N_INIT = 100
DT = 0.15
SPEED_MAX = 8.0
VISION = 180.0  # 視野角
R_SENSE = 180.0 # 視程
R_HIT = 12.0
FOOD_RATE = 0.01   # ステップあたりスポーン確率
FOOD_EN = 50.0
DECAY_BODY = 0.995

# MLP：入力dimはセンサー×本数＋自己状態
IN_DIM = 5*3 + 4  # [距離, 相対サイズ, 相対速度投影]*5 + self(4)
H_DIM = 16
OUT_DIM = 5  # thrust, turn, attack, mate, eat_strength

rng = np.random.default_rng(0)

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

def mutate(p, sigma=0.05):
    q = {k: v.copy() for k,v in p.items()}
    for k in q:
        q[k] += rng.normal(0, sigma, q[k].shape)
    return q

class Agent:
    def __init__(self):
        self.x = rng.uniform(0, W)
        self.y = rng.uniform(0, H)
        ang = rng.uniform(0, 2*np.pi)
        self.vx, self.vy = np.cos(ang), np.sin(ang)
        self.S = rng.uniform(0.8, 1.2)     # 体格
        self.E = 200.0                     # エネルギー
        self.brain = mlp_init(IN_DIM, H_DIM, OUT_DIM)

    def sense(self, agents):
        # 簡易：前方±60°に5本の等角レイを近似（本実装では空間索引で最近接探索を）
        angles = np.linspace(-np.deg2rad(60), np.deg2rad(60), 5)
        theta = np.arctan2(self.vy, self.vx)
        feats = []
        for a in angles:
            dirx, diry = np.cos(theta+a), np.sin(theta+a)
            best_d = R_SENSE
            best_S = 0.0
            best_vproj = 0.0
            for other in agents:
                if other is self: continue
                dx = wrap(other.x - self.x, W)
                dy = wrap(other.y - self.y, H)
                d = np.hypot(dx, dy)
                if d < best_d:
                    # 方向一致度チェック（視野内）
                    dir2 = (dx/d, dy/d)
                    if dir2[0]*dirx + dir2[1]*diry > np.cos(np.deg2rad(90)):
                        best_d = d
                        best_S = other.S / self.S
                        relv = ((other.vx - self.vx)*dirx + (other.vy - self.vy)*diry)
                        best_vproj = relv
            feats += [best_d/R_SENSE, best_S, np.tanh(best_vproj/5.0)]
        # 自己状態
        spd = np.hypot(self.vx, self.vy)/SPEED_MAX
        feats += [self.E/400.0, self.S/1.5, spd, 0.0]  # 角速度は省略
        return np.array(feats, dtype=np.float32)

    def step(self, agents):
        x = self.sense(agents)
        o = mlp_forward(self.brain, x)
        thrust = np.tanh(o[0])            # [-1,1]
        turn = np.tanh(o[1]) * 0.3        # rad/step
        attack = o[2] > 0.5
        mate   = o[3] > 0.5
        eat_strength = np.clip(o[4], 0, 1)

        # 運動
        theta = np.arctan2(self.vy, self.vx) + turn
        spd = np.clip(np.hypot(self.vx, self.vy) + thrust, 0, SPEED_MAX*(1.2 - 0.2*self.S))
        self.vx, self.vy = spd*np.cos(theta), spd*np.sin(theta)
        self.x = wrap(self.x + self.vx*DT, W)
        self.y = wrap(self.y + self.vy*DT, H)

        # コスト
        move_cost = 0.002 * spd*spd
        brain_cost = 0.0002 * param_count(self.brain)
        self.E -= (move_cost + brain_cost)

        return attack, mate, eat_strength

def wrap(v, L):
    if v < 0: return v + L
    if v >= L: return v - L
    return v

def param_count(p): return sum(v.size for v in p.values())

# --- 環境 ---
agents = [Agent() for _ in range(N_INIT)]
food = []        # (x,y,energy)
bodies = []      # 死体：(x,y,energy)

def spawn_food():
    if rng.random() < FOOD_RATE:
        food.append([rng.uniform(0,W), rng.uniform(0,H), FOOD_EN])

for t in range(20000):
    spawn_food()

    # 行動フェーズ
    intents = []
    for a in agents:
        intents.append(a.step(agents))

    # 相互作用：捕食/交配/採餌（O(N^2)簡略。実装では空間分割で高速化を）
    newborns = []
    removed = set()
    for i,a in enumerate(agents):
        attack, mate, eat_str = intents[i]

        # 食べ物（環境餌）
        for f in food:
            if (abs(f[0]-a.x) < R_HIT) and (abs(f[1]-a.y) < R_HIT):
                if np.hypot(f[0]-a.x, f[1]-a.y) < R_HIT:
                    a.E += f[2] * (0.3 + 0.7*eat_str)
                    f[2] = 0
        # 死体スカベンジ
        for b in bodies:
            if np.hypot(b[0]-a.x, b[1]-a.y) < R_HIT:
                take = min(b[2], 10.0*(0.3 + 0.7*eat_str))
                a.E += take
                b[2] -= take

        # 対個体
        for j,b in enumerate(agents):
            if i==j or j in removed: continue
            if np.hypot(a.x-b.x, a.y-b.y) < R_HIT:
                # 交配
                if mate and intents[j][1]:  # 双方が交配意図
                    if a.E>220 and b.E>220 and is_compatible(a,b):
                        child = Agent()
                        child.brain = crossover(a.brain, b.brain)
                        child.brain = mutate(child.brain, sigma=0.08)
                        child.x, child.y = a.x, a.y
                        child.E = 120.0
                        a.E -= 80.0; b.E -= 80.0
                        newborns.append(child)
                # 攻撃
                if attack:
                    atk = 8.0*a.S
                    defn = 5.0*b.S
                    if atk > defn:
                        a.E += 60.0
                        removed.add(j)
                        bodies.append([b.x,b.y, 80.0])

    # 片付け
    agents = [a for k,a in enumerate(agents) if k not in removed and a.E>0]
    # 死体減衰 & 食糧整理
    for b in bodies: b[2] *= DECAY_BODY
    bodies[:] = [b for b in bodies if b[2] > 1.0]
    food[:] = [f for f in food if f[2] > 1.0]

    agents.extend(newborns)

    # 絶滅リセット（安全弁）
    if not agents:
        agents = [Agent() for _ in range(N_INIT)]

    # 観察用の簡易ログ
    if t % 500 == 0:
        meanE = np.mean([a.E for a in agents])
        print(f"t={t} n={len(agents)} meanE={meanE:.1f}")

# --- 交配判定などの補助 ---
def flat_params(p): return np.concatenate([p["W1"].ravel(), p["b1"], p["W2"].ravel(), p["b2"]])

def is_compatible(a, b, alpha=0.5, th=1.5):
    # 重み距離 + （簡易）速度行動距離
    d_w = np.linalg.norm(flat_params(a.brain) - flat_params(b.brain)) / 200.0
    d_beh = np.linalg.norm(np.array([a.vx,a.vy]) - np.array([b.vx,b.vy]))/SPEED_MAX
    return alpha*d_w + (1-alpha)*d_beh < th

def crossover(pa, pb):
    qa = {k: v.copy() for k,v in pa.items()}
    for k in qa:
        mask = rng.random(qa[k].shape) < 0.5
        qa[k][mask] = pb[k][mask]
    return qa