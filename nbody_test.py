import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib

# 设置支持中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 常量定义
G = 4 * np.pi ** 2  # 天文单位制下的引力常数 (AU^3 / yr^2 / M_sun)
M_sun = 1.0  # 太阳质量
M_jupiter = 1e-3  # 木星质量 (~1/1000太阳质量)
M_earth = 3e-6  # 地球质量 (~1/300000太阳质量)
t_stop = 100  # 阻力时间尺度 (yr)

# 初始条件
a_jupiter = 1.0  # 木星轨道半径 (AU)
a_earth = 2.0  # 地球轨道半径 (AU)
v_jupiter = np.sqrt(G * M_sun / a_jupiter)
v_earth_stable = np.sqrt(G * M_sun / a_earth)

v_earth = 0
v_angle = 0  # 地球初始速度角度
v_hw = 0  # 可修改气体速度偏移

vx_e = v_earth * np.cos(v_angle)
vy_e = v_earth_stable + v_earth * np.sin(v_angle)

# 初始状态向量 [x, y, vx, vy] * 2
y0 = [
    a_jupiter, 0.0, 0.0, v_jupiter,  # 木星
    a_earth, 0.0, vx_e, vy_e  # 地球
]


# 阻力力函数，仅作用于地球
def drag_force(v, v_gas):
    return -(v - v_gas) / t_stop


# 微分方程
def equations(t, y):
    x_j, y_j, vx_j, vy_j, x_e, y_e, vx_e, vy_e = y
    dx = x_e - x_j
    dy = y_e - y_j
    r_je = np.sqrt(dx ** 2 + dy ** 2)
    r_j = np.sqrt(x_j ** 2 + y_j ** 2)
    r_e = np.sqrt(x_e ** 2 + y_e ** 2)
    ax_j = -G * M_sun * x_j / r_j ** 3 + G * M_earth * dx / r_je ** 3
    ay_j = -G * M_sun * y_j / r_j ** 3 + G * M_earth * dy / r_je ** 3
    ax_e = -G * M_sun * x_e / r_e ** 3 - G * M_jupiter * dx / r_je ** 3
    ay_e = -G * M_sun * y_e / r_e ** 3 - G * M_jupiter * dy / r_je ** 3
    v_k = np.sqrt(G * M_sun / r_e)
    v_gas = (v_k - v_hw)
    v_e = np.array([vx_e, vy_e])
    e_phi = np.array([-y_e, x_e]) / r_e
    e_phi /= np.linalg.norm(e_phi)
    v_gas_vector = v_gas * e_phi
    a_drag = drag_force(v_e, v_gas_vector)
    ax_e += a_drag[0]
    ay_e += a_drag[1]
    return [vx_j, vy_j, ax_j, ay_j, vx_e, vy_e, ax_e, ay_e]


# 时间范围
t_span = (0, 40000)  # 总时间 (yr)
t_eval = np.linspace(*t_span, 500)  # 时间步

# 求解微分方程
sol = solve_ivp(equations, t_span, y0, t_eval=t_eval, rtol=1e-9, atol=1e-9)

# 提取结果
x_j = sol.y[0]
y_j = sol.y[1]
x_e = sol.y[4]
y_e = sol.y[5]

# 计算地球轨道半长轴
a_e = np.sqrt(x_e ** 2 + y_e ** 2)

# 绘制地球轨道半径随时间的变化
plt.figure(figsize=(10, 6))
plt.plot(sol.t, a_e)
plt.xlabel('时间 (yr)')
plt.ylabel('地球轨道半径 (AU)')
plt.title('地球轨道半径随时间的演化')
plt.grid()
plt.savefig(f"orbit_radius_v_earth_{v_earth}_v_angle_{v_angle}_v_hw_{v_hw}.png", dpi=300)
plt.show()

# 共振角计算
lambda_j = np.arctan2(y_j, x_j)  # 木星的平均经度
lambda_e = np.arctan2(y_e, x_e)  # 地球的平均经度

# 定义共振条件
resonances = [(2, 1), (3, 2), (5, 3), (7, 5)]  # 定义共振 p:q
phi_list = []

for p, q in resonances:
    phi = p * lambda_e - q * lambda_j  # 共振角公式
    phi = np.mod(phi + np.pi, 2 * np.pi) - np.pi  # 限制到 [-π, π]
    phi_list.append((p, q, phi))

# 共振检测函数，只检测最后1000年
resonance_threshold = 2  # 共振检测阈值 (rad)


def detect_resonance(phi, threshold, t, last_t=1000):
    # 获取最后1000年对应的数据
    t_end = t[-1]
    t_mask = t > (t_end - last_t)
    phi_recent = phi[t_mask]

    # 检测这些数据的最小和最大值
    phi_min = np.min(phi_recent)
    phi_max = np.max(phi_recent)
    if phi_max - phi_min < threshold:
        return True
    return False

# 分析共振捕获
# 计算地球和木星的轨道周期比率
T_j = 1.0  # 木星轨道周期 (yr)
T_e = a_e ** (1.5)  # 地球轨道周期 (根据开普勒第三定律)
# 计算周期比率
period_ratio = T_e / T_j
# 共振条件列表，如 2:1, 3:2 等
resonances = [2 / 1, 3 / 2, 5 / 3, 7 / 5]
# 绘制周期比率随时间的变化
plt.figure(figsize=(10, 6))
plt.plot(sol.t, period_ratio, label='周期比率 T_e / T_j')
# 绘制共振比率线
for res in resonances:
    plt.hlines(res, sol.t[0], sol.t[-1], linestyles='dashed', colors='r', alpha=0.5)
    plt.text(sol.t[-1], res + 0.02,
             f'{int(res)}:{int(1)} 共振' if res.is_integer() else f'{int(res * resonances[0])}:{int(resonances[0])} 共振',
             fontsize=10, color='r', ha='right')
plt.xlabel('时间 (yr)')
plt.ylabel('周期比率')
plt.title('周期比率随时间的演化')
plt.legend()
plt.grid()
plt.savefig(f"period_ratio_v_earth_{v_earth}_v_angle_{v_angle}_v_hw_{v_hw}.png", dpi=300)
plt.show()

# 筛选并绘制共振角
plt.figure(figsize=(12, 8))

for p, q, phi in phi_list:
    if detect_resonance(phi, resonance_threshold, sol.t):
        plt.plot(sol.t, phi, label=f'{p}:{q} 共振角')

plt.axhline(0, color='k', linestyle='dashed', linewidth=0.8)
plt.xlabel('时间 (yr)')
plt.ylabel('共振角 (rad)')
plt.title(f'共振角随时间的演化 (检测阈值: {resonance_threshold} rad)')
plt.legend()
plt.grid()
plt.savefig(f"resonance_detected_v_earth_{v_earth}_v_angle_{v_angle}_v_hw_{v_hw}.png", dpi=300)
plt.show()
