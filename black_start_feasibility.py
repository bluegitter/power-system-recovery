import numpy as np
from scipy.integrate import quad, fixed_quad
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

# 核密度估计
def kernel_density_estimate(data, bandwidth, x_grid):
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(data[:, np.newaxis])
    log_pdf = kde.score_samples(x_grid[:, np.newaxis])
    pdf = np.exp(log_pdf)
    return pdf

# 计算发电量 Q1 和需求电量 Q2
def calculate_energy(P, t1, t2, P_min):
    Q1 = np.trapz(P, dx=(t2-t1)/len(P))  # 数值积分计算发电量
    Q2 = P_min * (t2 - t1)  # 需求电量
    return Q1, Q2

# 计算可执行倾度 β1 和 β2
def calculate_tilt_degree(Q1, Q2, f, P_min, P_max):
    β1 = Q1 / Q2

    # 计算 P1 和 P2，增加积分点数量
    P1, _ = quad(f, P_min, P_max, limit=1000)
    P2, _ = quad(f, 0, P_min, limit=1000)

    # 或者使用 fixed_quad 进行积分
    # P1, _ = fixed_quad(f, P_min, P_max, n=100)
    # P2, _ = fixed_quad(f, 0, P_min, n=100)

    β2 = P1 / P2

    # 综合可执行倾度 β
    β = 0.5 * β1 + 0.5 * β2
    return β1, β2, β

# 修改后的示例数据
P = np.array([18, 20, 22, 25, 28, 30, 27, 26, 24, 23])  # 调整后的发电功率数据
t1, t2 = 0, 10  # 时间区间
P_min = 20  # 最小功率需求
bandwidth = 1.0  # KDE 带宽

# 计算发电量和需求电量
Q1, Q2 = calculate_energy(P, t1, t2, P_min)

# KDE 概率密度估计
x_grid = np.linspace(0, max(P), 1000)
pdf = kernel_density_estimate(P, bandwidth, x_grid)
f = lambda x: np.interp(x, x_grid, pdf)  # 概率密度函数

# 计算可执行倾度
β1, β2, β = calculate_tilt_degree(Q1, Q2, f, P_min, max(P))

print(f"发电量 Q1: {Q1:.2f}")
print(f"需求电量 Q2: {Q2:.2f}")
print(f"相对发电量的可执行倾度 β1: {β1:.2f}")
print(f"概率分布的可执行倾度 β2: {β2:.2f}")
print(f"综合可执行倾度 β: {β:.2f}")

# 绘制概率密度函数
plt.plot(x_grid, pdf, label='Probability Density Function')
plt.axvline(P_min, color='r', linestyle='--', label='P_min')
plt.xlabel('Power Output')
plt.ylabel('Density')
plt.legend()
plt.show()
