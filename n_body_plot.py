import numpy as np
from scipy.integrate import solve_ivp
import time
import matplotlib.pyplot as plt
import matplotlib


def detect_resonance_plot(v_earth, v_hw, v_angle):
    # 固定使用的参数
    a_earth = 2.0
    t_stop = 100
    M_jupiter = 1e-3
    max_time = 300  # 最大运行时长(秒)

    # 常量定义
    G = 4 * np.pi ** 2
    M_sun = 1.0
    a_jupiter = 1.0
    v_jupiter = np.sqrt(G * M_sun / a_jupiter)
    v_earth_stable = np.sqrt(G * M_sun / a_earth)

    # 初始状态向量
    vx_e = v_earth * np.cos(v_angle)
    print("vx_e =", vx_e)
    vy_e = v_earth_stable + v_earth * np.sin(v_angle)
    print("vy_e =", vy_e)
    y0 = [
        a_jupiter, 0.0, 0.0, v_jupiter,  # 木星
        a_earth, 0.0, vx_e, vy_e  # 地球
    ]

    def drag_force(v, v_gas):
        return -(v - v_gas) / t_stop

    def equations(t, y):
        x_j, y_j, vx_j, vy_j, x_e, y_e, vx_e, vy_e = y
        dx = x_e - x_j
        dy = y_e - y_j
        r_je = np.sqrt(dx ** 2 + dy ** 2)
        r_j = np.sqrt(x_j ** 2 + y_j ** 2)
        r_e = np.sqrt(x_e ** 2 + y_e ** 2)

        # 木星受太阳引力
        ax_j = -G * M_sun * x_j / r_j ** 3
        ay_j = -G * M_sun * y_j / r_j ** 3

        # 地球受太阳和木星引力
        ax_e = -G * M_sun * x_e / r_e ** 3 - G * M_jupiter * dx / r_je ** 3
        ay_e = -G * M_sun * y_e / r_e ** 3 - G * M_jupiter * dy / r_je ** 3

        # 计算气体速度 (简化为轨道速度 v_k 减去 v_hw)
        v_k = np.sqrt(G * M_sun / r_e)
        v_gas = v_k - v_hw
        v_e = np.array([vx_e, vy_e])
        # 近似气体速度方向为沿轨道方向 e_phi
        e_phi = np.array([-y_e, x_e]) / r_e
        e_phi /= np.linalg.norm(e_phi)
        a_drag = drag_force(v_e, v_gas * e_phi)

        ax_e += a_drag[0]
        ay_e += a_drag[1]
        return [vx_j, vy_j, ax_j, ay_j, vx_e, vy_e, ax_e, ay_e]

    def check_resonance(period_ratio):
        resonances = [2 / 1, 3 / 2, 5 / 3, 7 / 5]
        tolerance = 0.03
        for res in resonances:
            indices = np.where(np.abs(period_ratio - res) < tolerance)[0]
            if len(indices) > 10:
                nearby = period_ratio[indices]
                diff = np.diff(nearby)
                # 判断是否既非全递增也非全递减，可当作“振荡”
                is_increasing = np.all(diff >= 0)
                is_decreasing = np.all(diff <= 0)
                if not (is_increasing or is_decreasing):
                    return True
        return False

    # 设置中文字体
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    # 记录开始时间
    start_time = time.time()

    # 分段积分设置
    t0 = 0
    max_t = 25000
    segment_length = 1000
    current_state = y0

    # 用于存储绘图与分析数据
    t_points = []
    x_e_list = []
    y_e_list = []

    while t0 < max_t:
        # 超时检查
        if time.time() - start_time > max_time:
            print(f"\n计算超时 ({max_time}秒)，已计算至 {t0:.1f} 年")
            return False, None

        print(f"正在计算: {t0:.1f}/{max_t} 年 ({100 * t0 / max_t:.1f}%)", end='\n')
        t1 = min(t0 + segment_length, max_t)

        # 分段积分
        sol = solve_ivp(
            equations,
            (t0, t1),
            current_state,
            t_eval=np.linspace(t0, t1, 50),
            rtol=1e-6, atol=1e-6,
            max_step=10.0
        )

        if not sol.success:
            print(f"\n数值积分失败，已计算至 {t0:.1f} 年")
            return False, None

        # 记录数据
        t_points.extend(sol.t)
        x_e_list.extend(sol.y[4])
        y_e_list.extend(sol.y[5])

        # 在该段末尾判断是否发生共振
        x_e_chunk = sol.y[4]
        y_e_chunk = sol.y[5]
        a_e_chunk = np.sqrt(x_e_chunk ** 2 + y_e_chunk ** 2)
        T_j = 1.0
        T_e_chunk = a_e_chunk ** 1.5
        period_ratio_chunk = T_e_chunk / T_j

        if check_resonance(period_ratio_chunk):
            print(f"\n在 {t1:.1f} 年前后检测到可能的共振")

            # 绘制轨道 XY 演化
            plt.figure(figsize=(7, 5))
            plt.plot(x_e_list, y_e_list, label='地球轨道')
            plt.scatter(0, 0, c='orange', label='太阳')
            plt.xlabel('X (AU)')
            plt.ylabel('Y (AU)')
            plt.title('地球轨道演化')
            plt.legend()
            plt.grid()
            plt.show()

            # 绘制半长轴随时间变化
            # 半长轴就是r = sqrt(x^2 + y^2)这里假设接近轨道半长轴
            r_list = np.sqrt(np.array(x_e_list) ** 2 + np.array(y_e_list) ** 2)
            plt.figure(figsize=(7, 5))
            plt.plot(t_points, r_list, label='a(t)')
            plt.xlabel('时间 (yr)')
            plt.ylabel('地球轨道半长轴 (AU)')
            plt.title('地球轨道半长轴随时间演化')
            plt.legend()
            plt.grid()
            plt.show()

            return True, (np.array(t_points), np.array(x_e_list), np.array(y_e_list))

        # 准备下一段
        current_state = sol.y[:, -1]
        t0 = t1

    print(f"\n计算结束，在 {max_t} 年内未检测到明显共振")

    # 绘制最终轨道 XY 演化
    plt.figure(figsize=(7, 5))
    plt.plot(x_e_list, y_e_list, label='地球轨道')
    plt.scatter(0, 0, c='orange', label='太阳')
    plt.xlabel('X (AU)')
    plt.ylabel('Y (AU)')
    plt.title('地球轨道演化')
    plt.legend()
    plt.grid()
    plt.show()

    # 绘制最终半长轴随时间变化
    r_list = np.sqrt(np.array(x_e_list) ** 2 + np.array(y_e_list) ** 2)
    plt.figure(figsize=(7, 5))
    plt.plot(t_points, r_list, label='a(t)')
    plt.xlabel('时间 (yr)')
    plt.ylabel('地球轨道半长轴 (AU)')
    plt.title('地球轨道半长轴随时间演化')
    plt.legend()
    plt.grid()
    plt.show()

    return False, (np.array(t_points), np.array(x_e_list), np.array(y_e_list))