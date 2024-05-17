import pulp

# 定义测试数据
n_units = 3  # 机组数量
n_buses = 3  # 母线数量
n_lines = 3  # 线路数量
n_loads = 3  # 负荷数量
T = 4  # 时间段数量
ΔT = 1  # 每个时间段

# 定义负荷恢复情况的范围 (最小值和最大值)
P_L_min = [10, 20, 30]
P_L_max = [50, 60, 70]

# 定义目标函数中的 P_L,i^d
P_L = [
    [10, 20, 30, 40],
    [20, 30, 40, 50],
    [30, 40, 50, 60]
]

# 初始化模型
model = pulp.LpProblem("PowerSystemRecoveryOptimization", pulp.LpMaximize)

# 定义变量
k = pulp.LpVariable.dicts("k", (range(n_units), range(T)), 0, 1, pulp.LpBinary)
α = pulp.LpVariable.dicts("α", (range(n_buses), range(T)), 0, 1, pulp.LpBinary)
β = pulp.LpVariable.dicts("β", (range(n_lines), range(T)), 0, 1, pulp.LpBinary)
γ = pulp.LpVariable.dicts("γ", (range(n_loads), range(T)), 0, 1, pulp.LpBinary)
P_Li = pulp.LpVariable.dicts("P_Li", (range(n_loads), range(T)), 0, None)

# 目标函数
model += pulp.lpSum([P_L[i][d] * γ[i][d] * ΔT for i in range(n_loads) for d in range(T)])

# 约束条件
# 1. 机组启动约束
for i in range(n_units):
    for t in range(T - 1):
        model += k[i][t + 1] >= k[i][t]

# 2. 母线启动/停止约束
for i in range(n_buses):
    for t in range(T - 1):
        model += α[i][t + 1] >= α[i][t]

# 3. 线路恢复约束
for i in range(n_lines):
    for t in range(T - 1):
        model += β[i][t + 1] >= β[i][t]

# 4. 负荷恢复状态约束
for i in range(n_loads):
    for t in range(T - 1):
        model += γ[i][t + 1] >= γ[i][t]
        model += P_Li[i][t] >= P_L_min[i]
        model += P_Li[i][t] <= P_L_max[i]

# 5. 机组启动数量限制约束
for t in range(T):
    model += pulp.lpSum([k[i][t] for i in range(n_units)]) <= 1

# 6. 连接性次序性约束
for j in range(n_units):
    for i in range(n_buses):
        for t in range(T):
            model += k[j][t] <= α[i][t]

# 7. 线路恢复约束
for i in range(n_lines):
    for t in range(T - 1):
        model += β[i][t + 1] <= α[i][t] + α[i][t]

# 8. 节点恢复约束
for i in range(n_buses):
    for t in range(T):
        model += α[i][t] <= pulp.lpSum([β[j][t] for j in range(n_lines)])

# 9. 母线和负荷连接性约束
for i in range(n_loads):
    for t in range(T):
        model += γ[i][t] <= α[i][t]

# 求解模型
model.solve()

# 打印结果
for v in model.variables():
    print(f"{v.name} = {v.varValue}")

print(f"Total load recovered: {pulp.value(model.objective)}")
