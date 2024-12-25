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
# 计算初始速度
v_jupiter = np.sqrt(G * M_sun / a_jupiter)
v_earth_stable = np.sqrt(G * M_sun / a_earth)


v_earth = 0
v_angle = 0  # 地球初始速度角度
vx_e = v_earth * np.cos(v_angle)
vy_e = v_earth_stable + v_earth * np.sin(v_angle)

v_hw = 0  # 可修改气体速度偏移

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
    # 解包变量
    x_j, y_j, vx_j, vy_j, x_e, y_e, vx_e, vy_e = y
    # 计算距离
    dx = x_e - x_j
    dy = y_e - y_j
    r_je = np.sqrt(dx ** 2 + dy ** 2)
    r_j = np.sqrt(x_j ** 2 + y_j ** 2)
    r_e = np.sqrt(x_e ** 2 + y_e ** 2)
    # 木星受到的引力（考虑太阳和地球）
    ax_j = -G * M_sun * x_j / r_j ** 3
    ay_j = -G * M_sun * y_j / r_j ** 3
    # 地球受到的引力（考虑太阳和木星）
    ax_e = -G * M_sun * x_e / r_e ** 3 - G * M_jupiter * dx / r_je ** 3
    ay_e = -G * M_sun * y_e / r_e ** 3 - G * M_jupiter * dy / r_je ** 3
    # 计算气体速度
    v_k = np.sqrt(G * M_sun / r_e)
    v_gas = (v_k - v_hw)
    # 地球的速度向量和单位切向量
    v_e = np.array([vx_e, vy_e])
    e_phi = np.array([-y_e, x_e]) / r_e
    e_phi /= np.linalg.norm(e_phi)
    # 气体速度向量
    v_gas_vector = v_gas * e_phi
    # 阻力加速度
    a_drag = drag_force(v_e, v_gas_vector)
    # 将阻力加速度加到地球上
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
# 计算半长轴
a_e = np.sqrt(x_e ** 2 + y_e ** 2)
# 绘制地球轨道半径随时间的变化
plt.figure(figsize=(10, 6))
plt.plot(sol.t, a_e)
plt.xlabel('时间 (yr)')
plt.ylabel('地球轨道半径 (AU)')
plt.title('地球轨道半径随时间的演化')
plt.grid()
plt.show()
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
plt.ylabel('周期比率 T_e / T_j')
plt.title('轨道周期比率随时间的演化')
plt.legend()
plt.grid()
plt.show()
# 判断共振捕获
# 设置共振判断的容差
tolerance = 0.03
for res in resonances:
    resonance_indices = np.where(np.abs(period_ratio - res) < tolerance)[0]
    if len(resonance_indices) > 0:
        resonance_time = sol.t[resonance_indices[0]]
        print(f"发生 {int(res)}:{int(1)} 共振捕获的时间：{resonance_time:.2f} yr")
