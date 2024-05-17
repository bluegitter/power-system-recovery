import pulp

# 定义测试数据
n_loads = 3  # 负荷数量
n_sources = 2  # 电源数量
n_lines = 3  # 线路数量
T = 4  # 时间段数量
ΔT = 1  # 每个时间段

# 定义负荷恢复情况的范围 (最小值和最大值)
P_L_min = [10, 20, 30]
P_L_max = [50, 60, 70]
Q_L_min = [5, 10, 15]
Q_L_max = [25, 30, 35]

# 定义电源的出力范围 (最小值和最大值)
P_S_min = [0, 0]
P_S_max = [100, 100]
Q_S_min = [0, 0]
Q_S_max = [50, 50]

# 定义目标函数中的权重系数
γ = [1, 1.5, 2]

# 初始化模型
model = pulp.LpProblem("DistributionSystemRecoveryOptimization", pulp.LpMaximize)

# 定义变量
P_L = pulp.LpVariable.dicts("P_L", (range(n_loads), range(T)), 0, None)
Q_L = pulp.LpVariable.dicts("Q_L", (range(n_loads), range(T)), 0, None)
P_S = pulp.LpVariable.dicts("P_S", (range(n_sources), range(T)), 0, None)
Q_S = pulp.LpVariable.dicts("Q_S", (range(n_sources), range(T)), 0, None)
ω = pulp.LpVariable.dicts("ω", (range(n_loads), range(T)), 0, 1, pulp.LpBinary)

# 目标函数: 最大化恢复的负荷量
model += pulp.lpSum([P_L[j][t] * γ[j] * ΔT for j in range(n_loads) for t in range(T)])

# 约束条件
# 1. 负荷约束
for j in range(n_loads):
    for t in range(T):
        model += P_L[j][t] >= P_L_min[j] * ω[j][t]
        model += P_L[j][t] <= P_L_max[j] * ω[j][t]
        model += Q_L[j][t] >= Q_L_min[j] * ω[j][t]
        model += Q_L[j][t] <= Q_L_max[j] * ω[j][t]

# 2. 电源出力约束
for i in range(n_sources):
    for t in range(T):
        model += P_S[i][t] >= P_S_min[i]
        model += P_S[i][t] <= P_S_max[i]
        model += Q_S[i][t] >= Q_S_min[i]
        model += Q_S[i][t] <= Q_S_max[i]

# 3. 节点电压约束
# 定义电压变量
U = pulp.LpVariable.dicts("U", (range(n_loads + n_sources), range(T)), 0, None)
U_min = 0.95
U_max = 1.05

for i in range(n_loads + n_sources):
    for t in range(T):
        model += U[i][t] >= U_min ** 2
        model += U[i][t] <= U_max ** 2

# 4. 电流容量约束
I_max = 100
I = pulp.LpVariable.dicts("I", (range(n_lines), range(T)), 0, I_max ** 2)

# 5. 潮流约束 (简单示例，不包含复杂的电力潮流方程)
# 示例：假设每个负荷节点连接到一个电源节点
for j in range(n_loads):
    for t in range(T):
        model += P_L[j][t] <= P_S[j % n_sources][t]
        model += Q_L[j][t] <= Q_S[j % n_sources][t]

# 求解模型
model.solve()

# 打印结果
for v in model.variables():
    print(f"{v.name} = {v.varValue}")

print(f"Total load restored: {pulp.value(model.objective)}")
