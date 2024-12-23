import numpy as np
from scipy.integrate import solve_ivp
from timeout_decorator import timeout, TimeoutError

@timeout(300)  # 设置超时时间为 300 秒
def detect_resonance(v_earth, v_hw, v_angle):
    # 固定参数
    a_earth = 2.0
    t_stop = 100
    M_jupiter = 1e-3

    # 常量定义
    G = 4 * np.pi**2
    M_sun = 1.0
    a_jupiter = 1.0
    v_jupiter = np.sqrt(G * M_sun / a_jupiter)
    v_earth_stable = np.sqrt(G * M_sun / a_earth)

    # 初始状态向量
    vx_e = v_earth * np.cos(v_angle)
    vy_e = v_earth_stable + v_earth * np.sin(v_angle)
    y0 = [
        a_jupiter, 0.0, 0.0, v_jupiter,
        a_earth,   0.0, vx_e, vy_e
    ]

    def drag_force(v, v_gas):
        return -(v - v_gas) / t_stop

    def equations(t, y):
        x_j, y_j, vx_j, vy_j, x_e, y_e, vx_e, vy_e = y
        dx = x_e - x_j
        dy = y_e - y_j
        r_je = np.sqrt(dx**2 + dy**2)
        r_j = np.sqrt(x_j**2 + y_j**2)
        r_e = np.sqrt(x_e**2 + y_e**2)

        ax_j = -G * M_sun * x_j / r_j**3
        ay_j = -G * M_sun * y_j / r_j**3
        ax_e = -G * M_sun * x_e / r_e**3 - G * M_jupiter * dx / r_je**3
        ay_e = -G * M_sun * y_e / r_e**3 - G * M_jupiter * dy / r_je**3

        v_k = np.sqrt(G * M_sun / r_e)
        v_gas = v_k - v_hw
        v_e = np.array([vx_e, vy_e])
        e_phi = np.array([-y_e, x_e]) / r_e
        e_phi /= np.linalg.norm(e_phi)
        a_drag = drag_force(v_e, v_gas * e_phi)

        ax_e += a_drag[0]
        ay_e += a_drag[1]
        return [vx_j, vy_j, ax_j, ay_j, vx_e, vy_e, ax_e, ay_e]

    t_span = (0, 25000)
    t_eval = np.linspace(*t_span, 500)

    try:
        # 尝试解方程
        sol = solve_ivp(equations, t_span, y0, t_eval=t_eval, rtol=1e-7, atol=1e-7, max_step=10)

        # 提取部分或完整解
        x_e, y_e = sol.y[4], sol.y[5]
        a_e = np.sqrt(x_e**2 + y_e**2)
        T_j = 1.0
        T_e = a_e**1.5
        period_ratio = T_e / T_j

        # 检查共振
        return _check_resonance(period_ratio)

    except TimeoutError:
        print(f"运行超时：v_earth={v_earth}, v_hw={v_hw}, v_angle={v_angle}")

        # 如果超时，从已经计算的部分解中提取结果
        if 'sol' in locals() and sol.y.size > 0:
            x_e, y_e = sol.y[4], sol.y[5]
            a_e = np.sqrt(x_e**2 + y_e**2)
            T_j = 1.0
            T_e = a_e**1.5
            period_ratio = T_e / T_j

            # 使用部分解检查共振
            return _check_resonance(period_ratio)

        # 如果没有任何解，返回 None
        return None

def _check_resonance(period_ratio):
    resonances = [2/1, 3/2, 5/3, 7/5]
    tolerance = 0.03

    for res in resonances:
        indices = np.where(np.abs(period_ratio - res) < tolerance)[0]
        if len(indices) > 10:
            nearby = period_ratio[indices]

            # 检查是否非单调
            diff = np.diff(nearby)  # 计算相邻值的差分
            is_increasing = np.all(diff >= 0)  # 是否全递增
            is_decreasing = np.all(diff <= 0)  # 是否全递减

            if not (is_increasing or is_decreasing):  # 如果既不是递增也不是递减
                return True
    return False
