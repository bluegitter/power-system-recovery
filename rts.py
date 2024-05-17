import numpy as np
import cvxpy as cp

# 14节点数据
# name Pload Qload
bus = np.array([
    [1, 0, 0],
    [2, 21.7, 12.7],
    [3, 94.2, 19],
    [4, 47.8, -3.9],
    [5, 7.6, 1.6],
    [6, 11.2, 7.5],
    [7, 0, 0],
    [8, 0, 0],
    [9, 29.5, 16.6],
    [10, 3.5, 5.8],
    [11, 3.5, 1.8],
    [12, 6.1, 1.6],
    [13, 13.5, 5.8],
    [14, 14.9, 5]
])

# Gbus   Pmax    Ps  r    t2_1  t3  Pmin   Th
gen = np.array([
    [1, 332.4, 0, 1.11, 0, 0, 0, 900],
    [2, 140, 5, 1.37, 6, 11, 34.25, 900],
    [3, 100, 4, 1.27, 10, 15, 31.75, 900]
    # [6, 100, 3, 1.17, 8, 13, 29.25, 900],
    # [8, 100, 2.3, 0.87, 4, 10, 26.1, 900]
])

genwi6 = np.array([0.27, 0.27, 0.27, 0.38, 0.38, 0.38, 0.32, 0.32, 0.32, 0.49, 0.49, 0.49, 0.49, 0.49, 0.49, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.41, 0.41, 0.41])
genpv8 = np.array([0.96, 0.96, 0.96, 1.85, 1.85, 1.85, 2.44, 2.44, 2.44, 3.41, 3.41, 3.41, 3.74, 3.74, 3.74, 4.18, 4.18, 4.18, 4.11, 4.11, 4.11, 5.40, 5.40, 5.40])

# branch node1 node2
bra = np.array([
    [1, 1, 2],
    [2, 1, 5],
    [3, 2, 3],
    [4, 2, 4],
    [5, 2, 5],
    [6, 3, 4],
    [7, 4, 5],
    [8, 4, 7],
    [9, 4, 9],
    [10, 5, 6],
    [11, 6, 11],
    [12, 6, 12],
    [13, 6, 13],
    [14, 7, 8],
    [15, 7, 9],
    [16, 9, 10],
    [17, 9, 14],
    [18, 10, 11],
    [19, 12, 13],
    [20, 13, 14]
])

T = 120
dt = 5
Nt = T // dt

Nbus = bus.shape[0]
Nbra = bra.shape[0]
Ngen = gen.shape[0]

M = 9999

IM = np.zeros((Nbra, Nbus))
for l in range(Nbra):
    IM[l, bra[l, 1] - 1] = 1
    IM[l, bra[l, 2] - 1] = 1

t = np.ones(Nt * Ngen) * dt
tl = np.ones(Nt) * dt

#定义优化问题
# 定义优化变量
P = cp.Variable(Nt * Ngen)
a = cp.Variable((Nbus, Nt), boolean=True)
b = cp.Variable((Nbra, Nt), boolean=True)
k = cp.Variable((Ngen, Nt), boolean=True)
x = cp.Variable((Ngen, Nt), boolean=True)
y = cp.Variable((Ngen, Nt), boolean=True)
PL = cp.Variable((Nbus, Nt))
kk = cp.Variable((2, Nt), boolean=True)

# 定义目标函数
objective = cp.Maximize(cp.sum(P @ t) + genwi6 @ kk[0, :] * dt + genpv8 @ kk[1, :] * dt)

# 定义约束条件
constraints = []

# 连接性和次序约束
for tt in range(Nt - 1):
    g = 0
    gg = 0
    for i in range(Nbus):
        if i in [0, 1, 2]:  # 索引从0开始
            g += 1
            constraints.append(k[g - 1, tt] <= a[i, tt])
        elif i in [5, 7]:  # 索引从0开始
            gg += 1
            constraints.append(kk[gg - 1, tt] <= a[i, tt])
        constraints.append(a[i, tt + 1] <= b[:, tt + 1].T @ IM[:, i])
    for l in range(Nbra):
        constraints.append(b[l, tt + 1] <= a[bra[l, 1] - 1, tt] + a[bra[l, 2] - 1, tt])

# 其他约束
for tt in range(Nt - 1):
    for i in range(Nbus):
        constraints.append(a[i, tt] <= a[i, tt + 1])
    for g in range(Ngen):
        constraints.append(k[g, tt] <= k[g, tt + 1])
        constraints.append(x[g, tt] <= x[g, tt + 1])
        constraints.append(y[g, tt] <= y[g, tt + 1])
    for gg in range(2):
        constraints.append(kk[gg, tt] <= kk[gg, tt + 1])
    for l in range(Nbra):
        constraints.append(b[l, tt] <= b[l, tt + 1])

# 初始化约束
for i in range(Ngen):
    constraints.append(k[i, 0] == (1 if i == 0 else 0))
for i in range(1, Nbus):
    constraints.append(a[i, 0] == 0)
for l in range(Nbra):
    constraints.append(b[l, 0] == 0)
for gg in range(2):
    constraints.append(kk[gg, 0] == 0)

# 连接性次序约束中的约束
for tt in range(Nt - 1):
    scha = 0
    for l in range(Nbra):
        cha = b[l, tt + 1] - b[l, tt]
        scha += cha
    constraints.append(scha <= 1)

# 生成机组热启动时限约束
for i in range(Ngen):
    for tt in range(Nt):
        constraints.append((cp.sum(k[i, :tt + 1]) - gen[i, 4]) / T <= y[i, tt])
        constraints.append(y[i, tt] <= (cp.sum(k[i, :tt + 1]) - gen[i, 4] - 1) / T + 1)
        constraints.append((cp.sum(k[i, :tt + 1]) - gen[i, 5]) / T <= x[i, tt])
        constraints.append(x[i, tt] <= (cp.sum(k[i, :tt + 1]) - gen[i, 5] - 1) / T + 1)

# 生成机组功率和爬坡率约束
for i in range(Ngen):
    for tt in range(Nt):
        constraints.append(-k[i, tt] * M <= P[i + Ngen * (tt - 1)])
        constraints.append(P[i + Ngen * (tt - 1)] <= M * k[i, tt])
        constraints.append(P[i + Ngen * (tt - 1)] >= gen[i, 3] * (cp.sum(y[i, :tt + 1])) * dt - k[i, tt] * gen[i, 2] - x[i, tt] * M)
        constraints.append(P[i + Ngen * (tt - 1)] <= gen[i, 3] * (cp.sum(y[i, :tt + 1])) * dt - k[i, tt] * gen[i, 2] + x[i, tt] * M)
        constraints.append(-(1 - x[i, tt]) * M + gen[i, 6] - k[i, tt] * gen[i, 2] <= P[i + Ngen * (tt - 1)])
        constraints.append(P[i + Ngen * (tt - 1)] <= (1 - x[i, tt]) * M + gen[i, 1] - k[i, tt] * gen[i, 2])
        constraints.append(cp.sum(tt - k[i, :tt + 1]) + 1 <= gen[i, 7])

# 生成机组爬坡率约束
for i in range(Ngen):
    for tt in range(Nt - 1):
        constraints.append(-dt * gen[i, 3] <= P[i + Ngen * tt] - P[i + Ngen * (tt - 1)])
        constraints.append(P[i + Ngen * tt] - P[i + Ngen * (tt - 1)] <= dt * gen[i, 3])

# 系统功率平衡约束
for tt in range(Nt):
    for i in range(Nbus):
        if i == 10:  # 索引从0开始
            constraints.append(-3.5 * a[i, tt] <= PL[i, tt])
            constraints.append(PL[i, tt] <= a[i, tt] * bus[i, 1])
        else:
            constraints.append(0 <= PL[i, tt])
            constraints.append(PL[i, tt] <= a[i, tt] * bus[i, 1])
    constraints.append(cp.sum(P[Ngen * tt:Ngen * (tt + 1)]) + genwi6[tt] * kk[0, tt] + genpv8[tt] * kk[1, tt] - cp.sum(PL[:, tt]) == 0)

# 已恢复的负荷不再切除
for tt in range(Nt - 1):
    for i in range(Nbus):
        if i != 10:  # 索引从0开始
            constraints.append(PL[i, tt] <= PL[i, tt + 1])

# 定义和求解优化问题
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.MOSEK)

# 输出结果
print(f"Total load restored: {prob.value}")
for v in prob.variables():
    print(f"{v.name()} = {v.value}")


